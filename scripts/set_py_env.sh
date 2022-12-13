#!/bin/bash

DIR="$(pwd)"
export PYTHONPATH="${DIR}":$PYTHONPATH
echo "export PYTHONPATH=$DIR:\$PYTHONPATH"
