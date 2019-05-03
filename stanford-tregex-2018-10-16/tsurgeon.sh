#!/bin/sh

export CLASSPATH=/Users/brodieslab/Mood-Modality/stanford-tregex-2018-10-16/stanford-tregex.jar:$CLASSPATH
java -mx100m edu.stanford.nlp.trees.tregex.tsurgeon.Tsurgeon "$@"
