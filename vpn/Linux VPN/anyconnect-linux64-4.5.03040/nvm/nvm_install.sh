#!/bin/sh
#

BASH_BASE_SIZE=0x00000000
CISCO_AC_TIMESTAMP=0x0000000000000000
CISCO_AC_OBJNAME=1234567890123456789012345678901234567890123456789012345678901234
# BASH_BASE_SIZE=0x00000000 is required for signing
# CISCO_AC_TIMESTAMP is also required for signing
# comment is after BASH_BASE_SIZE or else sign tool will find the comment

TARROOT="nvm"
INSTPREFIX=/opt/cisco/anyconnect
BINDIR=${INSTPREFIX}/bin
LIBDIR=${INSTPREFIX}/lib
NVMDIR=${INSTPREFIX}/NVM
TMPNVMDIR=${INSTPREFIX}/NVM.tmp
PLUGINDIR=${BINDIR}/plugins
UNINST=${BINDIR}/nvm_uninstall.sh
VPNUNINSTALLSCRIPT=${BINDIR}/vpn_uninstall.sh
INSTALL=install
MARKER=$((`grep -an "[B]EGIN\ ARCHIVE" $0 | cut -d ":" -f 1` + 1))
MARKER_END=$((`grep -an "[E]ND\ ARCHIVE" $0 | cut -d ":" -f 1` - 1))
LOGFNAME=`date "+anyconnect-linux64-4.5.03040-nvm-install-%H%M%S%d%m%Y.log"`
CLIENTNAME="Cisco AnyConnect Network Visibility Client"
VPNMANIFEST="/opt/cisco/anyconnect/ACManifestVPN.xml"
KDFSRCTARFILE="ac_kdf_src.tar.gz"

echo "Installing ${CLIENTNAME}..."
echo "Installing ${CLIENTNAME}..." > /tmp/${LOGFNAME}
echo `whoami` "invoked $0 from " `pwd` " at " `date` >> /tmp/${LOGFNAME}

#Set a trap so that the log file is moved to ${INSTPREFIX}/. in any exit path
trap 'mv /tmp/${LOGFNAME} ${INSTPREFIX}/.' EXIT

# Make sure we are root
if [ `id | sed -e 's/(.*//'` != "uid=0" ]; then
  echo "Sorry, you need super user privileges to run this script."
  exit 1
fi

# NVM requires VPN to be installed. We check the presence of the vpn uninstall script to check if it is installed.
if [ ! -f ${VPNUNINSTALLSCRIPT} ]; then
    echo "VPN should be installed before NVM installation. Install VPN to proceed."
    echo "Exiting now."
    exit 1
fi

## The web-based installer used for VPN client installation and upgrades does
## not have the license.txt in the current directory, intentionally skipping
## the license agreement. Bug CSCtc45589 has been filed for this behavior.   
if [ -f "license.txt" ]; then
    cat ./license.txt
    echo
    echo -n "Do you accept the terms in the license agreement? [y/n] "
    read LICENSEAGREEMENT
    while : 
    do
      case ${LICENSEAGREEMENT} in
           [Yy][Ee][Ss])
                   echo "You have accepted the license agreement."
                   echo "Please wait while ${CLIENTNAME} is being installed..."
                   break
                   ;;
           [Yy])
                   echo "You have accepted the license agreement."
                   echo "Please wait while ${CLIENTNAME} is being installed..."
                   break
                   ;;
           [Nn][Oo])
                   echo "The installation was cancelled because you did not accept the license agreement."
                   exit 1
                   ;;
           [Nn])
                   echo "The installation was cancelled because you did not accept the license agreement."
                   exit 1
                   ;;
           *)    
                   echo "Please enter either \"y\" or \"n\"."
                   read LICENSEAGREEMENT
                   ;;
      esac
    done
fi
if [ "`basename $0`" != "nvm_install.sh" ]; then
  if which mktemp >/dev/null 2>&1; then
    TEMPDIR=`mktemp -d /tmp/nvm.XXXXXX`
    RMTEMP="yes"
  else
    TEMPDIR="/tmp"
    RMTEMP="no"
  fi
else
  TEMPDIR="."
fi

# In upgrades, build_ac_ko.sh & kdf src tar file should be deleted from NVM dir, 
# before moving NVM files to temp dir.
if [ -f ${NVMDIR}/${KDFSRCTARFILE} ]; then
  echo "rm -f ${NVMDIR}/${KDFSRCTARFILE}" >> /tmp/${LOGFNAME}
  rm -f ${NVMDIR}/${KDFSRCTARFILE} >> /tmp/${LOGFNAME} 2>&1
fi

if [ -f ${NVMDIR}/build_ac_ko.sh ]; then
  echo "rm -f ${NVMDIR}/build_ac_ko.sh" >> /tmp/${LOGFNAME}
  rm -f ${NVMDIR}/build_ac_ko.sh >> /tmp/${LOGFNAME} 2>&1
fi

# In upgrades, NVM files will be copied to a temp directory and moved back
if [ -d ${NVMDIR} ]; then
  echo "mv -f ${NVMDIR} ${TMPNVMDIR}" >> /tmp/${LOGFNAME}
  mv -f ${NVMDIR} ${TMPNVMDIR} >> /tmp/${LOGFNAME} 2>&1
fi

#
# Check for and uninstall any previous version.
#
if [ -x "${UNINST}" ]; then
  echo "Removing previous installation..."
  echo "Removing previous installation: ${UNINST}" >> /tmp/${LOGFNAME}
  if ! ${UNINST}; then
    echo "Error removing previous installation!  Continuing..."
    echo "Error removing previous installation!  Continuing..." >> /tmp/${LOGFNAME}
  fi
fi

if [ "${TEMPDIR}" != "." ]; then
  TARNAME=`date +%N`
  TARFILE=${TEMPDIR}/nvminst${TARNAME}.tgz

  echo "Extracting installation files to ${TARFILE}..."
  echo "Extracting installation files to ${TARFILE}..." >> /tmp/${LOGFNAME}
  # "head --bytes=-1" used to remove '\n' prior to MARKER_END
  head -n ${MARKER_END} $0 | tail -n +${MARKER} | head --bytes=-1 2>> /tmp/${LOGFNAME} > ${TARFILE} || exit 1

  echo "Unarchiving installation files to ${TEMPDIR}..."
  echo "Unarchiving installation files to ${TEMPDIR}..." >> /tmp/${LOGFNAME}
  tar xvzf ${TARFILE} -C ${TEMPDIR} >> /tmp/${LOGFNAME} 2>&1 || exit 1

  rm -f ${TARFILE}

  NEWTEMP="${TEMPDIR}/${TARROOT}"
else
  NEWTEMP="."
fi

# version of NVM being installed has to be same as installed VPN version

if [ -f "${NEWTEMP}/ACManifestNVM.xml" ] && [ -f ${VPNMANIFEST} ]; then
    VPNVERSION=$(sed -n '/VPNCore/{ s/<file .*version="//; s/".*$//; p; }' ${VPNMANIFEST})
    NVMVERSION=$(sed -n '/NVM/{ s/<file .*version="//; s/".*$//; p; }' "${NEWTEMP}/ACManifestNVM.xml")

    if [ ${VPNVERSION} != ${NVMVERSION} ]; then
        echo "Version ${NVMVERSION} of the Cisco AnyConnect VPN is required to install this package."
        echo "Version ${NVMVERSION} of the Cisco AnyConnect VPN is required to install this package." >> /tmp/${LOGFNAME}
        echo "Exiting now."
        echo "Exiting now." >> /tmp/${LOGFNAME}
        exit 1
    fi
fi

# build KDF first, if .ko doesn't exist.
cd ${NEWTEMP}
ACKDFKO="ac_kdf.ko"
if [ ! -f "${ACKDFKO}" ]; then
    echo "Starting to build AnyConnect Kernel Module..."
    echo "./build_ac_ko.sh build `pwd`" >> /tmp/${LOGFNAME}
    ./build_ac_ko.sh build `pwd` >> /tmp/${LOGFNAME} 2>&1
    if [ $? != 0 ]; then
        echo "Failed to build AnyConnect Kernel module."
        echo "Exiting now."
        exit 1
    else
        echo "AnyConnect Kernel module built successfully."
    fi
fi

# Return to previous directory.
cd -

# Make sure destination directories exist
# Since vpn installer creates these directories need to revisit

echo "Installing "${BINDIR} >> /tmp/${LOGFNAME}
${INSTALL} -d ${BINDIR} || exit 1
echo "Installing "${LIBDIR} >> /tmp/${LOGFNAME}
${INSTALL} -d ${LIBDIR} || exit 1
echo "Installing "${NVMDIR} >> /tmp/${LOGFNAME}
${INSTALL} -d ${NVMDIR} || exit 1
echo "Installing "${PLUGINDIR} >> /tmp/${LOGFNAME}
${INSTALL} -d ${PLUGINDIR} || exit 1

# Copy KDF source & build_ac_kdf_ko.sh to NVM dir.
if [ -d ${NVMDIR} ]; then
    echo "cp -af ${NEWTEMP}/${KDFSRCTARFILE} ${NVMDIR}" >> /tmp/${LOGFNAME}
    cp -af ${NEWTEMP}/${KDFSRCTARFILE} ${NVMDIR} >> /tmp/${LOGFNAME}

    echo "cp -af ${NEWTEMP}/build_ac_kdf_ko.sh ${NVMDIR}" >> /tmp/${LOGFNAME}
    cp -af ${NEWTEMP}/build_ac_ko.sh ${NVMDIR} >> /tmp/${LOGFNAME}
fi

echo "Installing "${NEWTEMP}/nvm_uninstall.sh >> /tmp/${LOGFNAME}
${INSTALL} -o root -m 755 ${NEWTEMP}/nvm_uninstall.sh ${BINDIR} || exit 1

echo "Installing "${NEWTEMP}/acnvmagent >> /tmp/${LOGFNAME}
${INSTALL} -o root -m 755 ${NEWTEMP}/acnvmagent ${BINDIR} || exit 1

echo "Installing "${NEWTEMP}/${ACKDFKO} >> /tmp/${LOGFNAME}
${INSTALL} -o root -m 755 ${NEWTEMP}/${ACKDFKO} ${BINDIR} || exit 1

echo "Installing "${NEWTEMP}/libsock_fltr_api.so >> /tmp/${LOGFNAME}
${INSTALL} -o root -m 755 ${NEWTEMP}/libsock_fltr_api.so ${LIBDIR} || exit 1

echo "Installing "${NEWTEMP}/plugins/libacnvmctrl.so >> /tmp/${LOGFNAME}
${INSTALL} -o root -m 755 ${NEWTEMP}/plugins/libacnvmctrl.so ${PLUGINDIR} || exit 1

if [ -f "${NEWTEMP}/ACManifestNVM.xml" ]; then
  echo "Installing "${NEWTEMP}/ACManifestNVM.xml >> /tmp/${LOGFNAME}
  ${INSTALL} -o root -m 444 ${NEWTEMP}/ACManifestNVM.xml ${INSTPREFIX} || exit 1
else
  echo "${NEWTEMP}/ACManifestNVM.xml does not exist. It will not be installed."
fi

# Generate/update the VPNManifest.dat file
if [ -f ${BINDIR}/manifesttool ]; then	
  ${BINDIR}/manifesttool -i ${INSTPREFIX} ${INSTPREFIX}/ACManifestNVM.xml
fi

if [ "${RMTEMP}" = "yes" ]; then
  echo rm -rf ${TEMPDIR} >> /tmp/${LOGFNAME}
  rm -rf ${TEMPDIR}
fi

# In upgrades, we restore the NVM directory from the temp dir
if [ -d ${TMPNVMDIR} ]; then
  echo "Moving NVM config files back to NVM directory" >> /tmp/${LOGFNAME}
  mkdir -p ${NVMDIR}
  tar cf - -C ${TMPNVMDIR} . | (cd ${NVMDIR}; tar xf -) >> /tmp/${LOGFNAME} 2>&1
  rm -rf ${TMPNVMDIR}
fi


echo "${CLIENTNAME} is installed successfully."
echo "${CLIENTNAME} is installed successfully." >> /tmp/${LOGFNAME}

exit 0

--BEGIN ARCHIVE--
