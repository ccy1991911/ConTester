echo -ne "\033]0;ENB\007"
cd lteENB
../../build/srsenb/src/srsenb ./enb.conf  --expert.lte_sample_rates=true
