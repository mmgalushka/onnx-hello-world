#!/bin/bash

# =============================================================================
# HELPER ACTIONS
# =============================================================================

NC=$(echo "\033[m")
BOLD=$(echo "\033[1;39m")
CMD=$(echo "\033[1;34m")
OPT=$(echo "\033[0;34m")

action_usage(){

                        
    echo -e "  ___  _   _ _   ___  __"
    echo -e " / _ \| \ | | \ | \ \/ /"
    echo -e "| | | |  \| |  \| |\  / "
    echo -e "| |_| | |\  | |\  |/  \ "
    echo -e " \___/|_| \_|_| \_/_/\_\\"
    echo -e "  Hello World           "                   
    echo -e ""
    echo -e "This is the project for everyone interested in learning about "
    echo -e "Open Neural Network Exchange (ONNX).                          "
    echo -e ""
    echo -e "ONNX is an open format built to represent machine learning    "
    echo -e "models. ONNX defines a common set of operators - the building "
    echo -e "blocks of machine learning and deep learning models - and a   "
    echo -e "common file format to enable AI developers to use models with "
    echo -e "a variety of frameworks, tools, runtimes, and compilers.      "
    echo -e ""
    echo -e "                                 Quote from: https://onnx.ai/ "

    echo -e "${BOLD}System Commands:${NC}"
    echo -e "   ${CMD}init${NC} initializers environment;"
    echo -e "   ${CMD}netron <ONNX FILE>${NC} launches model browser;" 
}

action_init(){
    if [ -d .venv ];
        then
            rm -r .venv
    fi

    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install pip --upgrade
    pip3 install setuptools --upgrade 
    pip3 install -r requirements.txt
}

action_netron(){
    source .venv/bin/activate
    netron -b ${@}
}

# =============================================================================
# HELPER COMMANDS SELECTOR
# =============================================================================
case $1 in
    init)
        action_init
    ;;
    netron)
        action_netron ${@:2}
    ;;
    *)
        action_usage
    ;;
esac  

exit 0