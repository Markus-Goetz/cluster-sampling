#!/usr/bin/env python

JOB_TEMPLATE = """#!/usr/bin/env bash

#MSUB -N cluster-sampling-{job}
#MSUB -l nodes=1:ppn={gpus}:visu
#MSUB -l walltime=03:00:00
#MSUB -m bae
#MSUB -M markus.goetz@kit.edu

newgrp fh2-project-devel
module restore py3
source ~/.virtualenvs/p3fh2/bin/activate

cd $PROJECT/cluster-sampling/cnn

{lines}

for pid in ${{processes[*]}};
do
    wait ${{pid}}
done
"""

LINE_TEMPLATE = """
export CUDA_VISIBLE_DEVICES={gpu}
./spectral_cnn.py -g -t -s {seed} -w 9 -b 50 -e 400  \\
--model "../out/{data}_{mask}_{seed}_model.h5" \\
--train-history "../out/{data}_{mask}_{seed}_train.csv" \\
--test-history "../out/{data}_{mask}_{seed}_test.csv" \\
--results "../out/{data}_{mask}_{seed}_results.csv" \\
../data/{data}.h5 ../data/{data}_{mask}.h5 &
processes[{gpu}]=$!
"""

GPUS = 4
DATA = 'houston'
MODES = ['random', 'size', 'stddev']
FRACTIONS = [0.1, 0.3, 0.6, 0.9]
SEEDS = range(5)
SAMPLING_PARAMETERS = {
    'random': range(5),
    'size': ['1.5_1', '1.5_9', '2.2_1', '2.2_9'],
    'stddev': ['1.5_1', '1.5_9', '2.2_1', '2.2_9']
}


def generate():
    i = 0
    lines = []

    for mode in MODES:
        for fraction in FRACTIONS:
            for seed in SEEDS:
                for parameters in SAMPLING_PARAMETERS[mode]:
                    gpu = i % GPUS
                    lines.append(LINE_TEMPLATE.format(
                        gpu=gpu,
                        data=DATA,
                        seed=seed,
                        mask='{}_{}_{}'.format(mode, fraction, parameters)
                    ))

                    if gpu == GPUS - 1:
                        job_id = i // GPUS
                        with open('submit_{}.sh'.format(job_id), 'w') as handle:
                            handle.write(JOB_TEMPLATE.format(
                                job=job_id,
                                gpus=GPUS,
                                lines=''.join(lines)
                            ))
                        lines = []
                    i += 1


if __name__ == '__main__':
    generate()
