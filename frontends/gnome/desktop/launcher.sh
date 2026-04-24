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

export PATH="$HOME/.local/bin:$PATH"
exec hugin-meet-recorder "$@"
