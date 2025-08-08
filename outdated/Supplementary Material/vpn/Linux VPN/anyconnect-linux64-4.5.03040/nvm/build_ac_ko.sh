#!/bin/bash

# Script to build NVM kernel module in /tmp dir.
# It will be called from "nvm_install.sh" script
# Note: This script should not be invoked independently.

ACKDFKO="ac_kdf.ko"
KDFSRCTARFILE="ac_kdf_src.tar.gz"
KDFDIR="kdf/lkm"
INSTPREFIX="/opt/cisco/anyconnect"
BINDIR="${INSTPREFIX}/bin"
RETVAL=0
CCFLAG=""

# Make sure we are root.
if [ `id | sed -e 's/(.*//'` != "uid=0" ]; then
  echo "You need super user privileges to run this script."
  exit 1
fi

# Check if valid parameter is passed.
if [ -z $1 ] || [ -z $2 ]; then
    echo "Usage: build_ac_ko.sh {build|update}"
    exit 1
fi

DIRTOCOPYACKDFKO=$2
 
# checks if kernel-headers/linux-headers pkg is installed.
isKernelHeaderInstalled()
{
    linux_kernel_header_pkg=0
    installed_linux_kernel_header_pkg=0
    KERNEL_VERSION=$(uname -r)

    if [ -f /etc/redhat-release ]; then
        linux_kernel_header_pkg=kernel-devel-${KERNEL_VERSION}
        installed_linux_kernel_header_pkg=`rpm -q kernel-devel-${KERNEL_VERSION}`
    else
        linux_kernel_header_pkg=linux-headers-${KERNEL_VERSION}
        installed_linux_kernel_header_pkg=`dpkg -s linux-headers-${KERNEL_VERSION} | grep Package | awk '{ print $2 }'`
    fi

    if [ "${linux_kernel_header_pkg}" != "${installed_linux_kernel_header_pkg}" ]; then
        echo "${linux_kernel_header_pkg} should be installed. Install ${linux_kernel_header_pkg} to proceed."
        echo "Exiting now."
        exit 1
    fi 
}

# Checks if installed gcc version is same with which kernel has been built.
# It checks against "major.minor" version of gcc release.
# e.g. For gccversion "4.4.x", it compares 4.4 and discards x.
checkGccVersion()
{
    gcc_in_proc_version=`cat /proc/version | awk -F'gcc version ' '{ print $2 }' | awk -F'.' '{ print $1"."$2 }'`
    proc_gcc_major_version_no=`echo ${gcc_in_proc_version} | awk -F'.' '{ print $1 }'`

    # First look for system's default gcc in /usr/bin dir to check,
    # If it's "major.minor" version is same as of gcc used to build kernel.
    if [ -f /usr/bin/gcc ]; then
        installed_gcc_version=`/usr/bin/gcc -dumpversion | awk -F'.' '{ print $1"."$2 }'`
        if [ "${installed_gcc_version}" == "${gcc_in_proc_version}" ]; then
            CCFLAG="/usr/bin/gcc"
        fi
    fi

    # Couldn't find /usr/bin/gcc same as of gcc used to build kernel.
    # looking for version specific gcc in /usr/bin/ dir.
    if [ -z ${CCFLAG} ] && [ -f /usr/bin/gcc"-${gcc_in_proc_version}" ]; then
        installed_gcc_version=`/usr/bin/gcc"-${gcc_in_proc_version}" -dumpversion | awk -F'.' '{ print $1"."$2 }'`
        if [ "${installed_gcc_version}" == "${gcc_in_proc_version}" ]; then
            CCFLAG="/usr/bin/gcc-${gcc_in_proc_version}"
        fi
    fi

    # In Ubuntu 16 version specific gcc naming convention is like "gcc-x"
    # So search finally for "gcc-x"
    if [ -z ${CCFLAG} ] && [ -f "/usr/bin/gcc-${proc_gcc_major_version_no}" ]; then
        installed_gcc_version=`/usr/bin/gcc"-${proc_gcc_major_version_no}" -dumpversion | awk -F'.' '{ print $1"."$2 }'`
        if [ "${installed_gcc_version}" == "${gcc_in_proc_version}" ]; then
            CCFLAG="/usr/bin/gcc-${proc_gcc_major_version_no}"
        fi
    fi

    # Couldn't find gcc installed in the system, which was used to build kernel.
    # Failure case.
    if [ -z ${CCFLAG} ]; then
        echo "gcc-${gcc_in_proc_version} should be installed. Install gcc-${gcc_in_proc_version} to proceed."
        echo "Exiting now."
        exit 1
    fi
}

# Checks if ac_kdf.ko is compatible with current kernel version.
isAcKoCompatible()
{
    NVMDIR="${INSTPREFIX}/NVM"

    if [ ! -f ${DIRTOCOPYACKDFKO}/${ACKDFKO} ]; then
        # NVM is not installed, do nothing.
        # return success.
        return 0
    fi

    # get vermagic value of ac_kdf.ko & kernel release.
    ACKDFKOVERSION=`modinfo -F vermagic ${DIRTOCOPYACKDFKO}/${ACKDFKO} | awk '{ print $1 }'`
    KERNELRELEASE=`uname -r`

    if [ "${ACKDFKOVERSION}" == "${KERNELRELEASE}" ]; then
        # AnyConnect Kernel module  is compatible with current kernel release.
        # return success.
        return 0
    fi
 
    # ac_kdf.ko is incompatible with current kernel version.
    # return failure.
    return 1
}

# Builds ac_kdf.ko, if it is not compatible with current kernel version.
updateAcKdfKo()
{
    # Check if ac_kdf.ko is compatible with current kernel version. 
    isAcKoCompatible
    if [ $? == 0 ]; then
        # AnyConnect Kernel module  is compatible with current kernel release.
        RETVAL=2
        return
    fi
 
    # Remove existing ac_kdf.ko and invoke kdf build script to create compatible ac_kdf.ko .
    rm -rf ${DIRTOCOPYACKDFKO}/${ACKDFKO}
 
    # make ac_kdf.ko module.
    cd ${NVMDIR}
    buildAcKdfKo
    if [ $? == 0 ]; then
        # AnyConnect kernel module is upgraded successfully.
        # return success
        return
    fi

    # Failed to upgrade ac_kdf.ko
    RETVAL=1
}

# Makes ac_kdf.ko module.
buildAcKdfKo()
{
    #locate a unique temp dir, to build kdf src.
    if which mktemp >/dev/null 2>&1; then
        TEMPDIR=`mktemp -d /tmp/lkm.XXXXXX`
    else
        TEMPDIR="/tmp"
    fi

    # copy kdf src in tmp dir & make.
    cp -af ${KDFSRCTARFILE} ${TEMPDIR} || exit 1

    cd ${TEMPDIR}
    tar -xvzf ${KDFSRCTARFILE} || exit 1
    cd ${KDFDIR}
 
    if make CC=${CCFLAG}; then
        # copy .ko in DIRTOCOPYACKDFKO dir.
        echo "Built AnyConnect kernel module successfully."
        cp -f ${ACKDFKO} ${DIRTOCOPYACKDFKO} || exit 1
        chmod 755 ${DIRTOCOPYACKDFKO}/${ACKDFKO}
    else
        echo "Failed to build AnyConnect Kernel module, exiting now."
        RETVAL=1
    fi
}

#check if GNU Make utility exists
if ! which make; then
    echo "GNU Make should be installed, Install GNU Make to proceed."
    echo "Exiting now."
    exit 1
fi

#check if kernel-header pkg installed.
isKernelHeaderInstalled

# check if installed gcc version is same with which kernel is built.
checkGccVersion

# See how we were called.
case "$1" in
  build)
    buildAcKdfKo
    ;;
  rebuild)
    updateAcKdfKo
    ;;
  *)
    echo "Usage: build_ac_ko.sh {build|update}"
    exit 1
esac

# cleanup
rm -rf ${TEMPDIR}

exit ${RETVAL}
