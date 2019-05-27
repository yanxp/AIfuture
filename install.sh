#!/bin/bash
# pip install -r requirements.txt
# conda env create -f environment.yaml
conda create -n afteam_911 python=3.6

# >>> conda init >>>

__conda_setup="$(CONDA_REPORT_ERRORS=false '$HOME/anaconda3/bin/conda' shell.bash hook 2> /dev/null)"

if [ $? -eq 0 ]; then

    \eval "$__conda_setup"

else

    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then

        . "$HOME/anaconda3/etc/profile.d/conda.sh"

        CONDA_CHANGEPS1=false conda activate base

    else

        \export PATH="$PATH:$HOME/anaconda3/bin"

    fi

fi

unset __conda_setup

# <<< conda init <<<

conda activate afteam_911

conda env update -f environment.yaml
