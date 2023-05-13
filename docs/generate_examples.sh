#!/bin/bash

echo "Generating HTML from examples..."
for f in examples/*; do
  pygmentize -f html -O style=pastie,linenos=1,full,wrapcode -l Pasp "$f" > "_site/${f%.*}.html"
done
echo "Done!"
