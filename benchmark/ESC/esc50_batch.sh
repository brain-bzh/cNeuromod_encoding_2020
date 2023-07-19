for LAYER in $(cat $1)
do
    for HID in $(cat $2)
    do

        for SEG in $(cat $3)
        do
            python baselin_ESC50.py --layer $LAYER --feat mean --hidden $HID --esc /home/nfarrugi/git/ESC-50 --seg $SEG --step 0.5 --padding 8 --dataset esc50 --save esc50_results.csv
        done
    done
done