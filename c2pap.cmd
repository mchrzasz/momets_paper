#!/bin/bash

# example:

# 10 chains
# ./c2pap.cmd 1 10 unfolding-flat-btokll

low=$1; shift
high=$1; shift
command=$1; shift

for i in $(seq -f %01.0f $low $high); do
    file=j${i}.job
    # beware of shell escaping: loadlever variables
    # must not be expanded by this shell
    echo "#! /bin/bash
#
#@ group = pr85tu
#@ job_type = serial
#@ class = serial
#@ node_usage = shared
#@ resources = ConsumableCpus(1)
#
###                    hh:mm:ss
#@ wall_clock_limit = 1:59:50
#@ job_name = mom\$(jobid)
#@ initialdir = \$(home)/workspace/papers/moments
#@ output = /gpfs/work/pr85tu/ru72xaf2/eos/moments/$command-$i.out
#@ error  = /gpfs/work/pr85tu/ru72xaf2/eos/moments/$command-$i.err
#@ notification=error
#@ notify_user=Frederik.Beaujean@lmu.de
#@ queue

python acceptance.py $command" > $file

    sync
    llsubmit $file
    rm $file
done
