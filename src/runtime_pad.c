/* libScePad on SDL3 gamepads, with a keyboard fallback. SDL events are pumped
 * by the window thread (gpu/shim/window.cpp); here state is only sampled.
 *
 * Bindings (keyboard, controller and mouse) come from input.ini, parsed by
 * runtime_input_config.c; see specs/keyboard-and-mouse/spec-design-keyboard-mouse-input.md.
 * Without an input.ini (a fresh install), the defaults it generates reproduce the previous
 * fixed layout, except: the keyboard now works together with a connected controller (MIX-001,
 * where it used to be keyboard-only-without-a-controller), Q is R3 (lock-on) instead of
 * Triangle, and C is also R3 (kept, so existing muscle memory for lock-on does not break):
 *   WASD left stick, arrow keys right stick, Space Cross, LShift Circle,
 *   E Square, V Triangle, 1 L1, 3 R1, R L2, F R2, Z L3, Q/C R3,
 *   Enter Options, Tab left touchpad, Backspace right touchpad,
 *   IJKL d-pad (I up, K down, J left, L right). */
#define _GNU_SOURCE
#include "runtime.h"
#include "gpu/bbgpu.h"
#include "runtime_input_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <math.h>
#include <SDL3/SDL.h>
#include <sys/stat.h>

#define ERR_INVALID_ARG ((int32_t)0x80920001)
#define ERR_INVALID_HANDLE ((int32_t)0x80920003)
#define ERR_ALREADY_OPENED ((int32_t)0x80920004)
#define ERR_NOT_INITIALIZED ((int32_t)0x80920005)
#define PAD_HANDLE 1

enum {
    BTN_L3=0x2, BTN_R3=0x4, BTN_OPTIONS=0x8, BTN_UP=0x10, BTN_RIGHT=0x20, BTN_DOWN=0x40, BTN_LEFT=0x80,
    BTN_L2=0x100, BTN_R2=0x200, BTN_L1=0x400, BTN_R1=0x800, BTN_TRIANGLE=0x1000, BTN_CIRCLE=0x2000,
    BTN_CROSS=0x4000, BTN_SQUARE=0x8000, BTN_TOUCHPAD=0x100000,
};
typedef struct { uint16_t x, y; uint8_t id, reserve[3]; } PadTouch;
typedef struct {
    uint32_t buttons;
    uint8_t left_x, left_y, right_x, right_y;
    uint8_t l2, r2, analog_padding[2];
    float orientation[4], acceleration[3], angular_velocity[3];
    uint8_t touch_count, touch_reserve[3];
    uint32_t touch_held_time;
    PadTouch touches[2];
    uint8_t connected, pad0[3];
    uint64_t timestamp;
    uint8_t extension[16];
    uint8_t connected_count, reserve[2], unique_length, unique[12];
} PadData;
typedef struct {
    float pixel_density; uint16_t resolution_x, resolution_y;
    uint8_t dead_zone_left, dead_zone_right, connection_type, connected_count;
    uint8_t connected, pad[3];
    int32_t device_class;
    uint8_t reserve[8];
} ControllerInfo;
_Static_assert(sizeof(PadData)==120,"OrbisPadData layout");
_Static_assert(sizeof(PadTouch)==8,"OrbisPadTouch layout");
_Static_assert(__builtin_offsetof(PadData,touches)==60,"OrbisPadData touch offset");
_Static_assert(__builtin_offsetof(PadData,timestamp)==80,"OrbisPadData timestamp offset");
_Static_assert(sizeof(ControllerInfo)==28,"OrbisPadControllerInformation layout");

static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static int initialized, opened, sdl_ready;
static SDL_Gamepad *gamepad;
static size_t reads;
static uint8_t connected_count;
static uint64_t last_mouse_sample_us; /* CNV-003: dt since the previous mouse-to-stick sample */
static double smoothed_mouse_stick_x, smoothed_mouse_stick_y; /* mouse_to_axis's EMA state */

static uint64_t now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000u+(uint64_t)t.tv_nsec/1000u; }
static uint8_t trigger(int16_t v) { int x=v>>7; return (uint8_t)(x<0 ? 0 : x>255 ? 255 : x); }
static uint16_t touch_axis(float v, int max) {
    return (uint16_t)(v<=0.0f ? 0 : v>=1.0f ? max : (int)(v*max+0.5f));
}
static void touch_click(PadData *d, int right) {
    d->buttons|=BTN_TOUCHPAD;
    d->touch_count=1;
    d->touches[0]=(PadTouch){.x=right ? 1440 : 480,.y=471,.id=0};
}

/* Opens the first gamepad SDL knows about; called under lock. */
static SDL_Gamepad *current_gamepad(void) {
    if (!sdl_ready) sdl_ready = SDL_WasInit(SDL_INIT_GAMEPAD) ? 1 : SDL_InitSubSystem(SDL_INIT_GAMEPAD) ? 1 : -1;
    if (sdl_ready<0) return NULL;
    if (gamepad && !SDL_GamepadConnected(gamepad)) { SDL_CloseGamepad(gamepad); gamepad=NULL; }
    if (!gamepad) {
        int count=0;
        SDL_JoystickID *ids=SDL_GetGamepads(&count);
        if (ids && count>0) {
            gamepad=SDL_OpenGamepad(ids[0]);
            if (gamepad) { ++connected_count; printf("Runtime: gamepad connected: %s\n",SDL_GetGamepadName(gamepad)); }
        }
        SDL_free(ids);
    }
    return gamepad;
}
/* Button-like outputs (not the stick/axis ones): the bit each sets in d->buttons, or 0 for
 * l2/r2 (handled separately, since they also carry an analog value) and for outputs that are
 * not plain buttons (touchpad sides, half-axes, halfmode, hotkeys). */
static uint32_t button_bit(InputOutput out) {
    switch (out) {
    case OUT_CROSS: return BTN_CROSS; case OUT_CIRCLE: return BTN_CIRCLE;
    case OUT_SQUARE: return BTN_SQUARE; case OUT_TRIANGLE: return BTN_TRIANGLE;
    case OUT_L1: return BTN_L1; case OUT_R1: return BTN_R1;
    case OUT_L3: return BTN_L3; case OUT_R3: return BTN_R3; case OUT_OPTIONS: return BTN_OPTIONS;
    case OUT_PAD_UP: return BTN_UP; case OUT_PAD_DOWN: return BTN_DOWN;
    case OUT_PAD_LEFT: return BTN_LEFT; case OUT_PAD_RIGHT: return BTN_RIGHT;
    default: return 0;
    }
}

/* Host state read once per sample() so every binding is evaluated against the same instant
 * (pressing a key exactly as SDL delivers the gamepad event would otherwise see different
 * states depending on which binding happens to be checked first). */
typedef struct {
    const bool *keys;      /* SDL_GetKeyboardState result, or NULL if no video subsystem */
    SDL_Gamepad *gamepad;   /* current_gamepad() result, or NULL if none connected */
    uint32_t mouse_buttons; /* bbgpu_mouse_take's buttons: SDL_BUTTON_MASK() bits (T5) */
    uint32_t mouse_wheel;   /* bbgpu_mouse_take's wheel: bit 0 up, 1 down, 2 left, 3 right */
} HostState;

/* True if one input source is currently active. Axis sources use the same thresholds as the
 * button readings below (OUT-004/OUT-005): a trigger above INPUT_TRIGGER_BUTTON_THRESHOLD, a
 * half/full stick axis deflected past INPUT_HALF_AXIS_BUTTON_THRESHOLD in its own direction. */
static int source_active(const HostState *host, const InputSource *s) {
    switch (s->kind) {
    case IN_KEY: return host->keys && host->keys[s->value];
    case IN_CBUTTON: return host->gamepad && SDL_GetGamepadButton(host->gamepad,(SDL_GamepadButton)s->value);
    case IN_AXIS: {
        /* OUT-004: a trigger (l2/r2) bound to a button-like output uses the 0..255 trigger
         * threshold. A full stick axis (axis_left_x etc.) bound to a button-like output is not
         * covered by name in the spec; treated like a half-axis (OUT-005's threshold) since it
         * is the same physical reading, just not split by sign at the binding site. */
        if (!host->gamepad) return 0;
        int16_t raw=SDL_GetGamepadAxis(host->gamepad,(SDL_GamepadAxis)s->value);
        if (s->value==SDL_GAMEPAD_AXIS_LEFT_TRIGGER || s->value==SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)
            return trigger(raw)>INPUT_TRIGGER_BUTTON_THRESHOLD;
        int v=raw>>8; /* -128..127 */
        return (s->half_sign>0 ? v : -v)>INPUT_HALF_AXIS_BUTTON_THRESHOLD;
    }
    case IN_AXIS_HALF: {
        if (!host->gamepad) return 0;
        int16_t raw=SDL_GetGamepadAxis(host->gamepad,(SDL_GamepadAxis)s->value);
        int v=raw>>8; /* -128..127 */
        return s->half_sign>0 ? v>INPUT_HALF_AXIS_BUTTON_THRESHOLD : -v>INPUT_HALF_AXIS_BUTTON_THRESHOLD;
    }
    /* MOU-006: mouse bindings only ever fire while bbgpu_mouse_take reported captured=1 --
     * enforced by sample_host() zeroing host->mouse_buttons/mouse_wheel outright when not
     * captured, so this function does not need to re-check capture itself. */
    case IN_MOUSE_BUTTON: return (host->mouse_buttons & SDL_BUTTON_MASK(s->value))!=0;
    case IN_MOUSE_WHEEL: return (host->mouse_wheel & (1u<<s->value))!=0;
    case IN_NONE: default: return 0;
    }
}
/* Evaluates one binding against the current host state (MIX-002/003/004): for a button-like
 * binding, 1 if held, 0 otherwise -- a combo needs every source held. (Axis-like readings of a
 * binding are binding_axis_value's job; `is_button_output` there tells which of the two the
 * caller wants, see OUT-004/OUT-005.) */
static int binding_held(const HostState *host, const InputBinding *b) {
    for (uint8_t i=0;i<b->key_count;++i)
        if (!source_active(host,&b->sources[i])) return 0;
    return b->key_count>0;
}
/* Signed axis-unit contribution (-127..127) of one binding toward the stick it is bound to;
 * 0 for anything that is not an axis-like input. A plain key/button bound to a half-axis
 * output contributes a full deflection (KBD-006), matching the previous fixed-layout behavior.
 * A combo contributes only as a whole (every source held); a lone physical axis keeps its
 * value even below the button threshold, so analog pass-through
 * (axis_left_x = axis_right_x) stays analog. */
static int binding_axis_value(const HostState *host, const InputBinding *b, int half_sign) {
    if (b->key_count>1 && !binding_held(host,b)) return 0;
    for (uint8_t i=0;i<b->key_count;++i) {
        const InputSource *s=&b->sources[i];
        switch (s->kind) {
        case IN_KEY: case IN_CBUTTON: case IN_MOUSE_BUTTON: case IN_MOUSE_WHEEL:
            return source_active(host,s) ? 127*half_sign : 0;
        case IN_AXIS_HALF: {
            if (!host->gamepad) return 0;
            int v=SDL_GetGamepadAxis(host->gamepad,(SDL_GamepadAxis)s->value)>>8;
            return s->half_sign>0 ? (v>0 ? v : 0) : (v<0 ? -v : 0);
        }
        case IN_AXIS: /* a full physical axis bound to another full axis output (axis_left_x=axis_right_x) */
            return host->gamepad ? (SDL_GetGamepadAxis(host->gamepad,(SDL_GamepadAxis)s->value)>>8)*half_sign : 0;
        default: break; /* IN_NONE */
        }
    }
    return 0;
}
/* shadPS4's binding pass (input_handler.cpp ProcessBinding): connections are evaluated
 * longest-binding-first and every input a fired binding used is claimed for that sample, so a
 * claimed input cannot satisfy another binding (shadPS4's pressed-keys flag). That is what
 * gives a combo priority over the single-key actions sharing its inputs: with shift+M1 held,
 * "r2 = lshift,leftbutton" fires while "r1 = leftbutton" and "leftjoystick_halfmode = lshift"
 * stay off, instead of all three firing together. Bindings made only of physical axes
 * (stick/trigger pass-throughs) take no part in the pass; they read their value directly, like
 * shadPS4's analog path. */
static uint8_t binding_fired[OUT_COUNT][INPUT_MAX_BINDINGS_PER_OUTPUT];

static int source_is_axis(const InputSource *s) {
    return s->kind==IN_AXIS || s->kind==IN_AXIS_HALF;
}
static int binding_is_axis_only(const InputBinding *b) {
    for (uint8_t i=0;i<b->key_count;++i)
        if (!source_is_axis(&b->sources[i])) return 0;
    return 1;
}
static int source_claimed(const InputSource *claimed, int count, const InputSource *s) {
    for (int i=0;i<count;++i)
        if (claimed[i].kind==s->kind && claimed[i].value==s->value &&
            (s->kind!=IN_AXIS_HALF || claimed[i].half_sign==s->half_sign)) return 1;
    return 0;
}
static void evaluate_bindings(const HostState *host, const InputConfig *cfg) {
    static InputSource claimed[3*OUT_COUNT*INPUT_MAX_BINDINGS_PER_OUTPUT]; /* one entry per source of every fired binding, at worst */
    int claimed_count=0;
    memset(binding_fired,0,sizeof(binding_fired));
    for (int want=3;want>=1;--want) {
        for (int out=0;out<OUT_COUNT;++out) {
            for (int i=0;i<cfg->table.binding_count[out];++i) {
                const InputBinding *b=&cfg->table.bindings[out][i];
                if (b->key_count!=want || binding_is_axis_only(b)) continue;
                int fire=1;
                for (uint8_t k=0;k<b->key_count && fire;++k) {
                    if (!source_active(host,&b->sources[k])) fire=0;
                    else if (source_claimed(claimed,claimed_count,&b->sources[k])) fire=0;
                }
                if (!fire) continue;
                for (uint8_t k=0;k<b->key_count;++k) claimed[claimed_count++]=b->sources[k];
                binding_fired[out][i]=1;
            }
        }
    }
}
/* True if `out` reads as pressed -- a fired binding, or an axis-only binding whose own
 * threshold is met (those bypass the pass above). */
static int output_button_held(const HostState *host, const InputConfig *cfg, InputOutput out) {
    for (int i=0;i<cfg->table.binding_count[out];++i) {
        const InputBinding *b=&cfg->table.bindings[out][i];
        if (binding_fired[out][i]) return 1;
        if (binding_is_axis_only(b) && binding_held(host,b)) return 1;
    }
    return 0;
}
/* Sums every fired binding on `out` (MIX-003), clamped to -127..127; axis-only bindings
 * contribute directly (analog pass-through). */
static int output_axis_value(const HostState *host, const InputConfig *cfg, InputOutput out, int half_sign) {
    long sum=0;
    for (int i=0;i<cfg->table.binding_count[out];++i) {
        const InputBinding *b=&cfg->table.bindings[out][i];
        if (!binding_fired[out][i] && !binding_is_axis_only(b)) continue;
        sum+=binding_axis_value(host,b,half_sign);
    }
    return sum<-127 ? -127 : sum>127 ? 127 : (int)sum;
}
/* DZN-001: shadPS4's ApplyDeadzone, ported -- below inner: 0; inner..outer: linear ramp to 127;
 * above outer: 127. Applied to the magnitude, sign preserved. */
static int apply_deadzone(int value, int inner, int outer) {
    int mag=abs(value);
    if (mag<=inner) return 0;
    if (mag>=outer) return value<0 ? -127 : 127;
    long scaled=(127L*(mag-inner))/(outer-inner);
    return value<0 ? -(int)scaled : (int)scaled;
}

/* STK-001: shadlixps4's UpdateAxisSmoothing, ported: a stick driven by keys/buttons ramps
 * linearly from where it was to its new target over 33 ms and snaps when within 8 units.
 * shadPS4 advances this on a 16 ms timer for every binding-driven axis (GameController::Axis,
 * smooth=true by default); here it advances once per sample, one game frame each.
 *
 * Without it a key press teleports the stick across in a single frame: releasing W as D goes
 * down snaps the stick direction 90 degrees at once, which the game turns into a hard corner;
 * with the ramp the value sweeps through the diagonal, so the character turns through it (the
 * "draws a circle" feel of shadPS4's keyboard sticks).
 *
 * The mouse contribution is added after the ramp and stays instant, matching shadPS4's
 * EmulateJoystick, whose axis writes pass smooth=false. */
#define STICK_RAMP_US 33000
typedef struct { int value, target, start; uint64_t start_us; } StickRamp;
static StickRamp stick_ramp[4]; /* left x, left y, right x, right y; zero = centered */
static int ramp_stick(StickRamp *r, int target, uint64_t now) {
    if (target != r->target) {
        r->target=target;
        r->start=r->value;
        r->start_us=now;
    }
    if (r->value != r->target) {
        if (abs(r->target-r->value)<8 || now-r->start_us>=STICK_RAMP_US) {
            r->value=r->target;
        } else {
            long t=(long)((now-r->start_us)*1000/STICK_RAMP_US);
            r->value=r->start+(int)((long)(r->target-r->start)*t/1000);
        }
    }
    return r->value;
}

/* MOU-009: converts raw mouse motion straight into shadPS4's linear response (magnitude*speed +
 * speed_offset*128, over a 33 ms window -- shadPS4's fixed poll interval, kept as the unit
 * mouse_movement_params' defaults are tuned for), with no filtering of the raw delta itself, and
 * then lerps the *output* stick value toward that target instead of jumping straight to it.
 * `smoothed_mouse_stick_x`/`smoothed_mouse_stick_y` (module-level state, same pattern as
 * `last_mouse_sample_us`, kept under their old names despite now holding a stick value rather
 * than a velocity) persist across calls and are reset to 0 whenever capture starts or ends --
 * MOU-005/CNV-002, see the caller in sample_host.
 *
 * History (four iterations, all found by playing, not by a test): the first port of shadPS4's
 * EmulateJoystick clamped speed up to a hard floor (deadzone_offset*128) whenever any motion was
 * seen -- a step function. At a high sample rate (this port's 150 FPS vs. shadPS4's fixed 33 ms/
 * ~30 Hz poll), a slow mouse move delivers tiny per-sample deltas that each jumped straight to
 * that floor and back to 0 the instant motion stopped: "micro-jumps". A second version replaced
 * the floor with a ramp scaling the linear response by each sample's *instantaneous* magnitude
 * (`1 - exp(-magnitude/k)`); that removed the magnitude=0 jump but was still stateless -- during
 * a fast 180-degree reversal, magnitude dips near zero at the inflection point (the hand
 * decelerates, reverses, accelerates), so the ramp reset mid-gesture and felt like the camera
 * stalled exactly when the reversal needed to be fastest; the same statelessness made even a
 * single fast flick from a standstill start slow, since the first sample always began the ramp
 * at 0. A third version filtered *velocity* (dx/dt, dy/dt) with a per-component EMA before the
 * formula, which fixed the reversal stall and slow start, but amplified the mouse's own sampling
 * noise: velocity is a derivative, and dividing small, irregular per-sample dt/dx by each other
 * at a high poll rate makes the filtered input itself noisy, which read as the camera starting
 * "too fast" and feeling jittery/"crispy" even after turning sensitivity down (the jitter isn't
 * in the sensitivity, it's upstream of it).
 *
 * This version moves the smoothing from input space (velocity) to output space (the stick value
 * the player actually perceives as camera rotation): the raw-delta formula runs unfiltered every
 * call (so it reacts immediately, with no reversal-stall or slow-start risk -- there is no
 * velocity history to reset), and only the final x/y stick value is lerped toward that target
 * with the same `alpha=1-exp(-dt_ms/tau_ms)` shape as before. This damps perceptual jitter
 * (whatever noise is in one sample's raw delta can only move the stick part-way toward it) without
 * reintroducing the reversal stall, since a lerp has no near-zero singularity to stall at -- it
 * just lerps straight through a sign change. `tau_ms=25` keeps a flick's 90%-of-target response
 * under 100 ms (imperceptible as lag) while still visibly damping single-pixel-level jitter at
 * high poll rates (tried 15 ms first; 25 ms cut the residual jitter spread by about a fifth for
 * under 35 ms of extra settle time, found by sweeping the constant and comparing both).
 * deadzone_offset no longer gates a floor or a ramp (there is no branch left to gate): it
 * is unused by this formula and kept in MouseParams only because mouse_movement_params still
 * carries three values in input.ini and the launcher still exposes a "Smoothness" slider for it
 * -- T5's follow-up may repurpose or retire it. */
static void mouse_to_axis(float dx, float dy, double dt_ms, const MouseParams *mouse, int *out_x, int *out_y) {
    const double tau_ms=25.0;
    double alpha=1.0-exp(-dt_ms/tau_ms);
    /* Scale the raw delta to "px per 33 ms" (shadPS4's fixed poll window) so the same physical
     * mouse speed produces the same magnitude regardless of this port's actual frame rate --
     * without this, a 150 FPS sample would see 1/4.5 of the delta a 33 ms/~30 Hz sample would for
     * the same hand motion, making sensitivity depend on FPS. This is a plain per-sample scale,
     * not a time filter, so it adds no state and no lag of its own. */
    double sdx=(double)dx*(33.0/dt_ms), sdy=(double)dy*(33.0/dt_ms);
    double magnitude=sqrt(sdx*sdx+sdy*sdy);
    double target_x=0.0, target_y=0.0;
    if (magnitude>=0.01) {
        /* speed_offset*128 is a sizeable minimum push (shadPS4's default, 0.125, is ~16 of 128
         * units) meant to raise the response for slow movement -- but switching it fully on/off
         * right at this magnitude>=0.01 boundary means the slightest jitter around that boundary
         * (the mouse delivering a 0 px sample, then a 1 px sample, then 0 again) toggles a 16-unit
         * contribution on and off every sample. The output lerp smooths that over several frames,
         * but doesn't remove it, since each toggle still perturbs the lerp's target. Fading
         * speed_offset in linearly over a short ramp (0 to ramp_mag) instead of switching it on
         * at full strength removes that remaining source of jitter without changing how a normal,
         * continuous slow movement feels (the ramp covers less than a pixel of travel). */
        const double ramp_mag=1.0;
        double offset_fade=magnitude<ramp_mag ? magnitude/ramp_mag : 1.0;
        double speed=magnitude*mouse->speed+mouse->speed_offset*128.0*offset_fade;
        if (speed>128.0) speed=128.0;
        double angle=atan2(sdy,sdx);
        target_x=cos(angle)*speed;
        target_y=sin(angle)*speed;
    }
    smoothed_mouse_stick_x+=(target_x-smoothed_mouse_stick_x)*alpha;
    smoothed_mouse_stick_y+=(target_y-smoothed_mouse_stick_y)*alpha;
    int x=(int)lround(smoothed_mouse_stick_x), y=(int)lround(smoothed_mouse_stick_y);
    *out_x=x<-127?-127:x>127?127:x;
    *out_y=y<-127?-127:y>127?127:y;
}

/* GUD-002: the loaded config is immutable once published; a reload (F8) builds a whole new one
 * off the pad lock (I/O stays off the hot path) and only the pointer swap happens under `lock`,
 * which already serializes every call into this file through pad_read_state. */
static InputConfig default_config;
static InputConfig *loaded_config; /* NULL until pad_open's first load; falls back to defaults */
static int config_initialized;
static void ensure_config_loaded(void) {
    if (config_initialized) return;
    config_initialized=1;
    input_config_defaults(&default_config);
    char path[1024];
    InputConfig *fresh=malloc(sizeof(*fresh));
    if (fresh && input_config_path(path,sizeof(path)) && input_config_load(fresh,path)) {
        loaded_config=fresh;
        if (fresh->warnings) printf("Input config: %s loaded, %u line(s) ignored (see above)\n",path,fresh->warnings);
        else printf("Input config: %s loaded\n",path);
    } else {
        free(fresh);
        puts("Input config: using built-in defaults");
    }
    bbgpu_input_configure(loaded_config ? loaded_config->mouse.stick!=0 : 0,
                          loaded_config ? loaded_config->toggle_scancode : default_config.toggle_scancode,
                          loaded_config ? loaded_config->reload_scancode : default_config.reload_scancode);
}
/* CFG-008: called once per sample() (under `lock`), reloads if the window thread set the F8
 * flag. The read of the new file (input_config_load, I/O) happens before the critical section
 * that follows would matter, but since sample() itself always runs under `lock` already, the
 * brief extra time here is the same class of work pad_read_state already does under that lock
 * every frame (SDL calls); a config file is small and local, unlike a blocking I/O wait. */
static void reload_config_if_requested(void) {
    if (!bbgpu_input_reload_requested()) return;
    char path[1024];
    InputConfig *fresh=malloc(sizeof(*fresh));
    if (!fresh) return;
    if (input_config_path(path,sizeof(path)) && input_config_load(fresh,path)) {
        InputConfig *old=loaded_config;
        loaded_config=fresh;
        free(old);
        if (fresh->warnings) printf("Input config: %s reloaded, %u line(s) ignored (see above)\n",path,fresh->warnings);
        else printf("Input config: %s reloaded\n",path);
        bbgpu_input_configure(loaded_config->mouse.stick!=0,loaded_config->toggle_scancode,loaded_config->reload_scancode);
    } else {
        free(fresh);
    }
}
/* CFG-003: loaded lazily on the first sample() after scePadOpen rather than inside pad_open
 * itself -- both run under `lock`, and the first call into sample_host() is effectively
 * "on open" from the game's point of view (nothing reads the pad before scePadOpen returns). */
static const InputConfig *active_config(void) {
    ensure_config_loaded();
    return loaded_config ? loaded_config : &default_config;
}

static void sample_host(PadData *d) {
    memset(d,0,sizeof(*d));
    d->left_x=d->left_y=d->right_x=d->right_y=128;
    d->orientation[3]=1.0f;
    d->connected=1; d->connected_count=connected_count ? connected_count : 1;
    d->timestamp=now_us();
    SDL_Gamepad *g=current_gamepad();
    if (bbgpu_overlay_captures_input()) return; /* settings menu open: neutral input */
    const InputConfig *cfg=active_config();
    BbMouseInput mouse; bbgpu_mouse_take(&mouse); /* MOU-008: drains dx/dy; buttons/wheel persist */
    HostState host={SDL_WasInit(SDL_INIT_VIDEO) ? SDL_GetKeyboardState(NULL) : NULL, g,
        mouse.captured ? mouse.buttons : 0, mouse.captured ? mouse.wheel : 0};
    evaluate_bindings(&host,cfg); /* one priority/claim pass per sample; output_* read it */

    /* MIX-001/002: keyboard, controller (and, from T5, mouse) are read together, buttons ORed.
     * OUT_L2/OUT_R2 are skipped here (button_bit returns 0 for them anyway) and handled below,
     * since they also carry an analog value that a plain button/key binding must snap to 255. */
    static const InputOutput plain_buttons[]={
        OUT_CROSS,OUT_CIRCLE,OUT_SQUARE,OUT_TRIANGLE,OUT_L1,OUT_R1,OUT_L3,OUT_R3,OUT_OPTIONS,
        OUT_PAD_UP,OUT_PAD_DOWN,OUT_PAD_LEFT,OUT_PAD_RIGHT,
    };
    for (size_t i=0;i<sizeof(plain_buttons)/sizeof(*plain_buttons);++i) {
        InputOutput out=plain_buttons[i];
        if (output_button_held(&host,cfg,out)) d->buttons|=button_bit(out);
    }
    /* OUT-003/OUT-004: l2/r2 carry both a button bit and an analog value; the contributions ADD
     * UP, like shadPS4's new_param accumulator -- an idle physical trigger bound to the same
     * output must never erase a keyboard/mouse press (with both "r2 = lshift,leftbutton" and
     * the controller's "r2 = r2" in the file, the combo holds R2 at 255 while the stick at rest
     * adds 0). A lone IN_AXIS source passes the physical reading through; a fired binding (or an
     * axis-only one whose threshold is met) adds a full press. */
    int32_t l2_value=0, r2_value=0;
    for (int i=0;i<cfg->table.binding_count[OUT_L2];++i) {
        const InputBinding *b=&cfg->table.bindings[OUT_L2][i];
        if (b->key_count==1 && b->sources[0].kind==IN_AXIS) {
            if (host.gamepad) l2_value+=trigger(SDL_GetGamepadAxis(host.gamepad,(SDL_GamepadAxis)b->sources[0].value));
        } else if (binding_fired[OUT_L2][i] || (binding_is_axis_only(b) && binding_held(&host,b))) l2_value+=255;
    }
    for (int i=0;i<cfg->table.binding_count[OUT_R2];++i) {
        const InputBinding *b=&cfg->table.bindings[OUT_R2][i];
        if (b->key_count==1 && b->sources[0].kind==IN_AXIS) {
            if (host.gamepad) r2_value+=trigger(SDL_GetGamepadAxis(host.gamepad,(SDL_GamepadAxis)b->sources[0].value));
        } else if (binding_fired[OUT_R2][i] || (binding_is_axis_only(b) && binding_held(&host,b))) r2_value+=255;
    }
    d->l2=l2_value>255 ? 255 : (uint8_t)l2_value;
    d->r2=r2_value>255 ? 255 : (uint8_t)r2_value;
    if (d->l2>INPUT_TRIGGER_BUTTON_THRESHOLD) d->buttons|=BTN_L2;
    if (d->r2>INPUT_TRIGGER_BUTTON_THRESHOLD) d->buttons|=BTN_R2;

    /* MIX-003/HLF-001/DZN-001: sticks, summed across sources, then deadzone, then halfmode. */
    int lx=output_axis_value(&host,cfg,OUT_AXIS_LEFT_X_MINUS,-1)+output_axis_value(&host,cfg,OUT_AXIS_LEFT_X_PLUS,1)
          +output_axis_value(&host,cfg,OUT_AXIS_LEFT_X,1);
    int ly=output_axis_value(&host,cfg,OUT_AXIS_LEFT_Y_MINUS,-1)+output_axis_value(&host,cfg,OUT_AXIS_LEFT_Y_PLUS,1)
          +output_axis_value(&host,cfg,OUT_AXIS_LEFT_Y,1);
    int rx=output_axis_value(&host,cfg,OUT_AXIS_RIGHT_X_MINUS,-1)+output_axis_value(&host,cfg,OUT_AXIS_RIGHT_X_PLUS,1)
          +output_axis_value(&host,cfg,OUT_AXIS_RIGHT_X,1);
    int ry=output_axis_value(&host,cfg,OUT_AXIS_RIGHT_Y_MINUS,-1)+output_axis_value(&host,cfg,OUT_AXIS_RIGHT_Y_PLUS,1)
          +output_axis_value(&host,cfg,OUT_AXIS_RIGHT_Y,1);
    lx=lx<-127?-127:lx>127?127:lx; ly=ly<-127?-127:ly>127?127:ly;
    rx=rx<-127?-127:rx>127?127:rx; ry=ry<-127?-127:ry>127?127:ry;
    lx=apply_deadzone(lx,cfg->deadzone[DEADZONE_LEFT_STICK].inner,cfg->deadzone[DEADZONE_LEFT_STICK].outer);
    ly=apply_deadzone(ly,cfg->deadzone[DEADZONE_LEFT_STICK].inner,cfg->deadzone[DEADZONE_LEFT_STICK].outer);
    rx=apply_deadzone(rx,cfg->deadzone[DEADZONE_RIGHT_STICK].inner,cfg->deadzone[DEADZONE_RIGHT_STICK].outer);
    ry=apply_deadzone(ry,cfg->deadzone[DEADZONE_RIGHT_STICK].inner,cfg->deadzone[DEADZONE_RIGHT_STICK].outer);

    /* HLF-001: half-speed (walk) halves the stick before the ramp, like shadPS4's multiplier
     * at the mapping step. The mouse below is outside the bindings pipeline and is not halved
     * (shadPS4's mouse writes bypass the multiplier the same way). */
    if (output_button_held(&host,cfg,OUT_LEFTJOYSTICK_HALFMODE)) { lx/=2; ly/=2; }
    if (output_button_held(&host,cfg,OUT_RIGHTJOYSTICK_HALFMODE)) { rx/=2; ry/=2; }

    uint64_t stick_now=now_us();
    lx=ramp_stick(&stick_ramp[0],lx,stick_now);
    ly=ramp_stick(&stick_ramp[1],ly,stick_now);
    rx=ramp_stick(&stick_ramp[2],rx,stick_now);
    ry=ramp_stick(&stick_ramp[3],ry,stick_now);

    /* CNV-001/MOU-009: the mouse contribution is added after the stick's own deadzone and ramp,
     * not before them -- mouse_to_axis's own output-value lerp already keeps small/slow motion
     * from producing a jumpy or oversized contribution (see its doc comment), so running the
     * result through apply_deadzone too would just chop off the small end of an already-smooth
     * curve, and adding it before the ramp would make it glide into the keyboard's target. */
    if (mouse.captured && cfg->mouse.stick) {
        uint64_t now=stick_now;
        double dt_ms=last_mouse_sample_us ? (double)(now-last_mouse_sample_us)/1000.0 : 33.0;
        if (dt_ms<1.0) dt_ms=1.0;
        if (dt_ms>100.0) dt_ms=100.0;
        last_mouse_sample_us=now;
        int mx=0, my=0;
        mouse_to_axis(mouse.dx,mouse.dy,dt_ms,&cfg->mouse,&mx,&my);
        int *tx=cfg->mouse.stick==1 ? &lx : &rx, *ty=cfg->mouse.stick==1 ? &ly : &ry;
        *tx+=mx; *ty+=my;
        *tx=*tx<-127?-127:*tx>127?127:*tx; *ty=*ty<-127?-127:*ty>127?127:*ty;
    } else {
        /* MOU-005/CNV-002: no stale dt or velocity-filter memory survives a capture gap -- the
         * next capture must start the EMA fresh, not carry over speed from before the gap. */
        last_mouse_sample_us=0;
        smoothed_mouse_stick_x=smoothed_mouse_stick_y=0.0;
    }

    d->left_x=(uint8_t)(128+lx); d->left_y=(uint8_t)(128+ly);
    d->right_x=(uint8_t)(128+rx); d->right_y=(uint8_t)(128+ry);

    /* OUT-002: the physical touchpad (real finger position/click) is unaffected by remap. */
    if (g) {
        if (SDL_GetGamepadButton(g,SDL_GAMEPAD_BUTTON_BACK) || SDL_GetGamepadButton(g,SDL_GAMEPAD_BUTTON_TOUCHPAD))
            d->buttons|=BTN_TOUCHPAD;
        if (SDL_GetNumGamepadTouchpads(g)>0) {
            const int fingers=SDL_GetNumGamepadTouchpadFingers(g,0);
            for (int finger=0;finger<fingers && d->touch_count<2;++finger) {
                bool down=false;
                float x=0, y=0;
                if (SDL_GetGamepadTouchpadFinger(g,0,finger,&down,&x,&y,NULL) && down) {
                    d->touches[d->touch_count++]=(PadTouch){.x=touch_axis(x,1919),
                        .y=touch_axis(y,942),.id=(uint8_t)finger};
                }
            }
        }
        // Back/Select on pads without a touch surface is a left-side click.
        if ((d->buttons & BTN_TOUCHPAD) && !d->touch_count) touch_click(d,0);
    }
    /* OUT-001: touchpad_left/center/right outputs, remappable, independent of the physical pad. */
    if (output_button_held(&host,cfg,OUT_TOUCHPAD_LEFT)) touch_click(d,0);
    if (output_button_held(&host,cfg,OUT_TOUCHPAD_RIGHT)) touch_click(d,1);
    if (output_button_held(&host,cfg,OUT_TOUCHPAD_CENTER)) { d->buttons|=BTN_TOUCHPAD; d->touch_count=1; d->touches[0]=(PadTouch){.x=960,.y=471,.id=0}; }
}

/* BB_PAD_FILE=<file>: scripted input for automated runs. The file holds whitespace-separated
 * tokens, re-read when it changes: button names (cross circle square triangle l1 r1 l2 r2 l3 r3
 * options touchpad touchpad_left touchpad_right up down left right) are held while listed;
 * touchpad defaults to a left-side click; lx= ly= rx= ry= (0..255) override
 * the sticks. An empty file releases everything. */
static struct { uint32_t buttons; int stick[4]; int touch_side; } injected={0,{-1,-1,-1,-1},-1};
static int replay_armed;      /* 1 while a BB_PAD_REPLAY recording plays, 2 once it ended */
static uint64_t replay_start; /* 0: (re)start at the next sample */
static void read_inject(void) {
    static const char *path; static int checked; static uint64_t last_check; static struct timespec mtime;
    if (!checked) { path=getenv("BB_PAD_FILE"); checked=1; }
    if (!path || !*path) return;
    uint64_t now=now_us();
    if (now-last_check<20000) return;
    last_check=now;
    struct stat st;
    if (stat(path,&st)!=0) return;
#ifdef _WIN32
    /* Whole-second mtimes: the size tells two edits within a second apart. */
    if (st.st_mtime==mtime.tv_sec && st.st_size==mtime.tv_nsec) return;
    mtime=(struct timespec){st.st_mtime,(long)st.st_size};
#else
    if (st.st_mtim.tv_sec==mtime.tv_sec && st.st_mtim.tv_nsec==mtime.tv_nsec) return;
    mtime=st.st_mtim;
#endif
    FILE *f=fopen(path,"r");
    if (!f) return;
    static const struct { const char *name; uint32_t ps; } names[]={
        {"cross",BTN_CROSS}, {"circle",BTN_CIRCLE}, {"square",BTN_SQUARE}, {"triangle",BTN_TRIANGLE},
        {"l1",BTN_L1}, {"r1",BTN_R1}, {"l2",BTN_L2}, {"r2",BTN_R2}, {"l3",BTN_L3}, {"r3",BTN_R3},
        {"options",BTN_OPTIONS}, {"touchpad",BTN_TOUCHPAD},
        {"up",BTN_UP}, {"down",BTN_DOWN}, {"left",BTN_LEFT}, {"right",BTN_RIGHT},
    };
    static const char *sticks[]={"lx=","ly=","rx=","ry="};
    injected.buttons=0;
    injected.touch_side=-1;
    for (int i=0;i<4;++i) injected.stick[i]=-1;
    char token[64];
    while (fscanf(f,"%63s",token)==1) {
        if (!strcmp(token,"replay") && replay_armed!=1) { replay_armed=1; replay_start=0; } /* BB_PAD_REPLAY */
        if (!strcmp(token,"touchpad_left") || !strcmp(token,"touchpad_right")) {
            injected.buttons|=BTN_TOUCHPAD;
            injected.touch_side=!strcmp(token,"touchpad_right");
        }
        for (size_t i=0;i<sizeof(names)/sizeof(*names);++i) if (!strcmp(token,names[i].name)) injected.buttons|=names[i].ps;
        for (int i=0;i<4;++i) if (!strncmp(token,sticks[i],3)) { int v=atoi(token+3); injected.stick[i]=v<0 ? 0 : v>255 ? 255 : v; }
    }
    fclose(f);
    printf("Runtime: pad file: buttons 0x%x sticks %d %d %d %d\n",injected.buttons,
           injected.stick[0],injected.stick[1],injected.stick[2],injected.stick[3]);
}
/* BB_PAD_RECORD=<file>: F9 starts and stops recording the pad state (gamepad or keyboard) with
 * the time since F9; BB_PAD_REPLAY=<file> plays such a recording back, started by the token
 * "replay" in BB_PAD_FILE (scripted tests repeat a route the player ran once). Lines: ms buttons
 * lx ly rx ry l2 r2, written when the state changes. */
typedef struct { uint32_t ms, buttons; uint8_t axes[4], l2, r2; } PadSample;
static FILE *record_file;
static uint64_t record_start;
static PadSample record_last;
static void record_sample(const PadData *d) {
    static const char *path; static int checked, f9_was_down;
    if (!checked) { path=getenv("BB_PAD_RECORD"); checked=1; }
    if (!path || !*path || !sdl_ready) return;
    const bool *k=SDL_GetKeyboardState(NULL);
    const int f9=k && k[SDL_SCANCODE_F9];
    if (f9 && !f9_was_down) {
        if (record_file) {
            fclose(record_file); record_file=NULL;
            printf("Runtime: pad recording stopped (%s)\n",path);
        } else if ((record_file=fopen(path,"w"))) {
            record_start=now_us();
            memset(&record_last,0xff,sizeof(record_last));
            printf("Runtime: pad recording started (%s, F9 stops)\n",path);
        }
    }
    f9_was_down=f9;
    if (!record_file) return;
    PadSample s={(uint32_t)((now_us()-record_start)/1000),d->buttons,
                 {d->left_x,d->left_y,d->right_x,d->right_y},d->l2,d->r2};
    if (s.buttons==record_last.buttons && !memcmp(s.axes,record_last.axes,4) &&
        s.l2==record_last.l2 && s.r2==record_last.r2) return;
    record_last=s;
    fprintf(record_file,"%u %u %u %u %u %u %u %u\n",s.ms,s.buttons,s.axes[0],s.axes[1],s.axes[2],
            s.axes[3],s.l2,s.r2);
    fflush(record_file);
}
static PadSample *replay; static size_t replay_count, replay_next;
static void replay_sample(PadData *d) {
    if (!replay_armed) return;
    if (!replay_start) {
        static int loaded;
        if (!loaded) {
            loaded=1;
            const char *path=getenv("BB_PAD_REPLAY");
            FILE *f=path ? fopen(path,"r") : NULL;
            PadSample s; unsigned v[8]; size_t cap=0;
            while (f && fscanf(f,"%u %u %u %u %u %u %u %u",&v[0],&v[1],&v[2],&v[3],&v[4],&v[5],&v[6],&v[7])==8) {
                s=(PadSample){v[0],v[1],{(uint8_t)v[2],(uint8_t)v[3],(uint8_t)v[4],(uint8_t)v[5]},(uint8_t)v[6],(uint8_t)v[7]};
                if (replay_count==cap && !(replay=realloc(replay,(cap=cap ? cap*2 : 1024)*sizeof(*replay)))) break;
                replay[replay_count++]=s;
            }
            if (f) fclose(f);
            printf("Runtime: pad replay of %zu samples from %s\n",replay_count,path ? path : "(unset)");
        }
        replay_start=now_us();
        replay_next=0;
    }
    const uint32_t ms=(uint32_t)((now_us()-replay_start)/1000);
    while (replay_next<replay_count && replay[replay_next].ms<=ms) ++replay_next;
    if (!replay_next) return;
    if (replay_next==replay_count && ms>replay[replay_count-1].ms+500) {
        if (replay_armed==1) { puts("Runtime: pad replay finished"); replay_armed=2; }
        return;
    }
    const PadSample *s=&replay[replay_next-1];
    d->buttons=s->buttons;
    d->left_x=s->axes[0]; d->left_y=s->axes[1]; d->right_x=s->axes[2]; d->right_y=s->axes[3];
    d->l2=s->l2; d->r2=s->r2;
}
/* After the menu or the text dialog closes, buttons still held (the Cross that accepted a
 * name) stay hidden until released: the game would take them as a new press. */
static int hold_after_capture;
static void sample(PadData *d) {
    reload_config_if_requested();
    sample_host(d);
    if (bbgpu_overlay_captures_input()) { hold_after_capture=1; return; }
    record_sample(d);
    read_inject();
    replay_sample(d);
    d->buttons|=injected.buttons;
    if (injected.touch_side>=0) touch_click(d,injected.touch_side);
    else if ((d->buttons & BTN_TOUCHPAD) && !d->touch_count) touch_click(d,0);
    if (injected.buttons & BTN_L2) d->l2=255;
    if (injected.buttons & BTN_R2) d->r2=255;
    uint8_t *axes[4]={&d->left_x,&d->left_y,&d->right_x,&d->right_y};
    for (int i=0;i<4;++i) if (injected.stick[i]>=0) *axes[i]=(uint8_t)injected.stick[i];
    if (hold_after_capture) {
        if (d->buttons) d->buttons=0;
        else hold_after_capture=0;
    }
}

static ABI int32_t pad_init(void) { pthread_mutex_lock(&lock); initialized=1; pthread_mutex_unlock(&lock); return 0; }
static ABI int32_t pad_open(int32_t user, int32_t type, int32_t index, const void *param) {
    (void)param;
    if (!initialized) return ERR_NOT_INITIALIZED;
    if (user!=1) return ERR_INVALID_ARG;
    if (type!=0 && type!=2) return ERR_INVALID_ARG; /* standard / special port */
    if (index) return ERR_INVALID_ARG;
    pthread_mutex_lock(&lock);
    int already=opened; opened=1;
    pthread_mutex_unlock(&lock);
    if (already) return ERR_ALREADY_OPENED;
    puts("Runtime: pad opened for user 1 (SDL gamepad or keyboard)");
    return PAD_HANDLE;
}
static ABI int32_t pad_close(int32_t handle) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    opened=0; return 0;
}
static ABI int32_t pad_read_state(int32_t handle, PadData *data) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!data) return ERR_INVALID_ARG;
    pthread_mutex_lock(&lock);
    sample(data); ++reads;
    pthread_mutex_unlock(&lock);
    return 0;
}
/* Buffered read: the port samples once per call, so one entry is returned. */
static ABI int32_t pad_read(int32_t handle, PadData *data, int32_t count) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!data || count<1 || count>64) return ERR_INVALID_ARG;
    pad_read_state(handle,data);
    return 1;
}
static ABI int32_t pad_info(int32_t handle, ControllerInfo *info) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!info) return ERR_INVALID_ARG;
    memset(info,0,sizeof(*info));
    info->pixel_density=44.86f; info->resolution_x=1920; info->resolution_y=943;
    info->dead_zone_left=info->dead_zone_right=2;
    info->connection_type=0; info->connected=1; info->device_class=0;
    pthread_mutex_lock(&lock);
    current_gamepad();
    info->connected_count=connected_count ? connected_count : 1;
    pthread_mutex_unlock(&lock);
    return 0;
}
static ABI int32_t pad_vibration(int32_t handle, const uint8_t *param) {
    if (handle!=PAD_HANDLE || !opened) return ERR_INVALID_HANDLE;
    if (!param) return ERR_INVALID_ARG;
    pthread_mutex_lock(&lock);
    SDL_Gamepad *g=current_gamepad();
    if (g) SDL_RumbleGamepad(g,(uint16_t)(param[0]*257),(uint16_t)(param[1]*257),1000);
    pthread_mutex_unlock(&lock);
    return 0;
}
static ABI int32_t pad_ok_handle(int32_t handle) { return handle==PAD_HANDLE && opened ? 0 : ERR_INVALID_HANDLE; }
static ABI int32_t pad_ok_handle_flag(int32_t handle, uint8_t flag) { (void)flag; return pad_ok_handle(handle); }

static const RuntimeExport exports[]={
    {"scePadInit",pad_init}, {"scePadOpen",pad_open}, {"scePadClose",pad_close},
    {"scePadReadState",pad_read_state}, {"scePadRead",pad_read},
    {"scePadGetControllerInformation",pad_info}, {"scePadSetVibration",pad_vibration},
    {"scePadResetOrientation",pad_ok_handle},
    {"scePadSetAngularVelocityDeadbandState",pad_ok_handle_flag}, {"scePadSetTiltCorrectionState",pad_ok_handle_flag},
    {"scePadSetMotionSensorState",pad_ok_handle_flag},
};
uintptr_t runtime_pad_resolve(const char *name) { return RUNTIME_LOOKUP(exports,name); }
void runtime_pad_report(void) { printf("Runtime: pad reads=%zu, gamepad=%s\n",reads,gamepad ? SDL_GetGamepadName(gamepad) : "none"); }
