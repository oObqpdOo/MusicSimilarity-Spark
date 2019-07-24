#!/bin/sh

INSTPREFIX="/opt/cisco/anyconnect"
BINDIR="${INSTPREFIX}/bin"
PLUGINSDIR="${BINDIR}/plugins"
LIBDIR="${INSTPREFIX}/lib"
NVMDIR="${INSTPREFIX}/NVM"
ACMANIFESTDAT="${INSTPREFIX}/VPNManifest.dat"
NVMMANIFEST="ACManifestNVM.xml"
UNINSTALLLOG="/tmp/nvm-uninstall.log"

# Array of files to remove
FILELIST="${INSTPREFIX}/${NVMMANIFEST} \
          ${INSTPREFIX}/libacnvmctrl.so \
          ${BINDIR}/acnvmagent \
          ${BINDIR}/ac_kdf.ko \
          ${LIBDIR}/libsock_fltr_api.so \
          ${BINDIR}/nvm_uninstall.sh"

echo "Uninstalling Cisco AnyConnect Network Visibility Module..."
echo "Uninstalling Cisco AnyConnect Network Visibility Module..." > ${UNINSTALLLOG}
echo `whoami` "invoked $0 from " `pwd` " at " `date` >> ${UNINSTALLLOG}

# Check for root privileges
if [ `whoami` != "root" ]; then
  echo "Sorry, you need super user privileges to run this script."
  echo "Sorry, you need super user privileges to run this script." >> ${UNINSTALLLOG}
  exit 1
fi

# update the VPNManifest.dat; if no entries remain in the .dat file then
# this tool will delete the file - DO NOT blindly delete VPNManifest.dat by
# adding it to the FILELIST above - allow this tool to delete the file if needed
if [ -f "${BINDIR}/manifesttool" ]; then
  echo "${BINDIR}/manifesttool -x ${INSTPREFIX} ${INSTPREFIX}/${NVMMANIFEST}" >> ${UNINSTALLLOG}
  ${BINDIR}/manifesttool -x ${INSTPREFIX} ${INSTPREFIX}/${NVMMANIFEST}
fi

# check the existence of the manifest file - if it does not exist, remove the manifesttool
if [ ! -f ${ACMANIFESTDAT} ] && [ -f ${BINDIR}/manifesttool ]; then
  echo "Removing ${BINDIR}/manifesttool" >> ${UNINSTALLLOG}
  rm -f ${BINDIR}/manifesttool
fi

# move the plugins to a different folder to stop the NVM agent and then remove
# these plugins once NVM agent is stopped. 
mv -f ${PLUGINSDIR}/libacnvmctrl.so ${INSTPREFIX} 2>&1 >/dev/null
echo "mv -f ${PLUGINSDIR}/libacnvmctrl.so ${INSTPREFIX}" >> ${UNINSTALLLOG}

# wait for 2 seconds for the NVM agent to exit
sleep 2

# ensure that the NVM agent is not running
NVMPROC=`ps -A -o pid,command | grep '(${BINDIR}/acnvmagent)' | egrep -v 'grep|nvm_uninstall' | awk '{print $1}'`
if [ ! "x${NVMPROC}" = "x" ] ; then
    echo Killing `ps -A -o pid,command -p ${NVMPROC} | grep ${NVMPROC} | egrep -v 'ps|grep'` >> ${UNINSTALLLOG}
    kill -TERM ${NVMPROC} >> ${UNINSTALLLOG} 2>&1
fi

# Remove the KDF
if /sbin/lsmod | grep ac_kdf > /dev/null; then
    echo "/sbin/rmmod -f ac_kdf" >> /tmp/nvm-uninstall.log
    /sbin/rmmod -f ac_kdf >> /tmp/nvm-uninstall.log 2>&1
fi

# Remove only those files that we know we installed
for FILE in ${FILELIST}; do
  echo "rm -f ${FILE}" >> /tmp/nvm-uninstall.log
  rm -f ${FILE} >> /tmp/nvm-uninstall.log 2>&1
done


# Remove the NVM directory
# During an upgrade, this dir(profile,cache,kconfig files) will be moved and restored by
# installer scripts

if [ -d ${NVMDIR} ]; then
    echo "rm -rf "${NVMDIR}"" >> ${UNINSTALLLOG}
    rm -rf "${NVMDIR}" >> ${UNINSTALLLOG} 2>&1
fi

echo "Successfully removed Cisco AnyConnect Network Visibility Module from the system." >> ${UNINSTALLLOG}
echo "Successfully removed Cisco AnyConnect Network Visibility Module from the system."

exit 0
