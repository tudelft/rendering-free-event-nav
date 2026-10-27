# Out-of-tree app build for the Crazyflie firmware.
#
# IMPORTANT: build for the BRUSHLESS platform with the DEFCONFIG, not PLATFORM=.
#   make cf21bl_defconfig   # sets CONFIG_PLATFORM_CF21BL=y + DSHOT (the real config)
#   make -j                 # build (produces build/cf21bl.bin)
#   make cload              # flash over the Crazyradio bootloader
#
# DO NOT use `make PLATFORM=cf21bl` - that only sets the output NAME/variable,
# not the config. Without cf21bl_defconfig it silently falls back to CF2
# (brushed, no DSHOT) and produces WRONG firmware for a brushless drone.
#
# CRAZYFLIE_BASE is a checkout of github.com/bitcraze/crazyflie-firmware.
# Default: cloned next to this repository. Override on the command line:
#   make CRAZYFLIE_BASE=/path/to/crazyflie-firmware cf21bl_defconfig

CRAZYFLIE_BASE ?= $(abspath $(CURDIR)/../../../crazyflie-firmware)

OOT_CONFIG := $(PWD)/app-config

include $(CRAZYFLIE_BASE)/tools/make/oot.mk
