#!/bin/sh

ANYCONNECT_BINDIR="/opt/cisco/anyconnect/bin"
POSTURE_BINDIR="/opt/cisco/hostscan/bin"

VPN_UNINST=${ANYCONNECT_BINDIR}/vpn_uninstall.sh
POSTURE_UNINST=${POSTURE_BINDIR}/posture_uninstall.sh
NVM_UNINST=${ANYCONNECT_BINDIR}/nvm_uninstall.sh

if [ -x "${POSTURE_UNINST}" ]; then
  ${POSTURE_UNINST}
  if [ $? -ne 0 ]; then
    echo "Error uninstalling AnyConnect Posture Module."
  fi
fi

if [ -x "${NVM_UNINST}" ]; then
  ${NVM_UNINST}
  if [ $? -ne 0 ]; then
  echo "Error uninstalling Network Visibility Module."
  fi
fi


if [ -x "${VPN_UNINST}" ]; then
  ${VPN_UNINST}
  if [ $? -ne 0 ]; then
    echo "Error uninstalling AnyConnect Secure Mobility Client."
  fi
fi

exit 0
