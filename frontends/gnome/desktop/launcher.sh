#!/bin/bash
# Launcher that cleans snap environment contamination before starting the recorder.
# VSCode snap sets GTK_PATH, GIO_MODULE_DIR etc. to snap paths, which pulls in
# old glibc and crashes. This wrapper unsets all of that.

unset GDK_PIXBUF_MODULEDIR GDK_PIXBUF_MODULE_FILE
unset GIO_MODULE_DIR GSETTINGS_SCHEMA_DIR
unset GTK_EXE_PREFIX GTK_IM_MODULE_FILE GTK_PATH
unset SNAP SNAP_ARCH SNAP_COMMON SNAP_CONTEXT SNAP_COOKIE
unset SNAP_DATA SNAP_EUID SNAP_INSTANCE_NAME SNAP_LAUNCHER_ARCH_TRIPLET
unset SNAP_LIBRARY_PATH SNAP_NAME SNAP_REAL_HOME SNAP_REVISION
unset SNAP_UID SNAP_USER_COMMON SNAP_USER_DATA SNAP_VERSION
unset LOCPATH

# If HUGIN_MEETINGS_VENV is set, prefer that venv so the tray and every
# subprocess it spawns use the same installed dependencies. Otherwise use the
# user's PATH, which works for user installs and packaged installs.
if [[ -n "${HUGIN_MEETINGS_VENV:-}" ]]; then
    VENV_BIN="$HUGIN_MEETINGS_VENV/bin"
fi

if [[ -n "${VENV_BIN:-}" && -x "$VENV_BIN/hugin-meet-recorder" ]]; then
    export VIRTUAL_ENV="$(dirname "$VENV_BIN")"
    export PATH="$VENV_BIN:$HOME/.local/bin:$PATH"
    # Clear PYTHONHOME/PYTHONPATH so the venv's site-packages isn't shadowed.
    unset PYTHONHOME PYTHONPATH
else
    if [[ -n "${VENV_BIN:-}" ]]; then
        echo "launcher.sh: no recorder in $VENV_BIN, falling back to ~/.local/bin" >&2
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

exec hugin-meet-recorder "$@"
