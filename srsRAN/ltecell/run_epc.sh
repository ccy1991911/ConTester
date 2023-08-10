echo -ne "\033]0;EPC\007"
cd lteEPC
sysctl -w net.ipv4.ip_forward=1
sync
iptables -t mangle -F FORWARD
iptables -t nat -F POSTROUTING
export LANG=C
#192.168.1.179 10.0.0.19 10.0.0.12
iptables -t nat -I POSTROUTING -s 172.16.0.0/12 -o wlp0s20f3 ! --protocol sctp -j SNAT --to-source 10.1.2.109
../../srsepc/srsepc_if_masq.sh wlp59s0
../../build/srsepc/src/srsepc epc.conf
