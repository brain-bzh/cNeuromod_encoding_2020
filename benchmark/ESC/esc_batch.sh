### tests on ESC10

for DATASET in $(cat $4)
do
    for SEGLEG in $(cat $1)
    do
        for STEP in $(cat $2)
        do
            for PAD in $(cat $3)
            do
                python baselin_ESC50.py --layer $5 --esc /home/nfarrugi/git/ESC-50 --dataset esc10 --seg $SEGLEG --step $STEP --padding $PAD --dataset $DATASET
            done
        done
    done
done