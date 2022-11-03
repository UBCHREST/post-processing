for dir in ../outputs
do
    for file in "$dir"/*.xml
    do
      	if [[ -f $file ]]
        then
            mv "$file" ../outputs/xmlFiles
        fi
    done
done