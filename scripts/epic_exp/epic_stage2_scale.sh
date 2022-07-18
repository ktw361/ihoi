for vframe in $(ls output/stage1/) 
do
    echo $vframe
    # testname="output/stage1/${vframe}"
    # OUT=$(find $testname -name 'clu*')
    # if [ -n "$OUT" ]; then
    #     echo "skip $testname"
    # else
    CUDA_VISIBLE_DEVICES=1 python demo/epic_stage2.py output/stage1/$vframe
    # fi
done
