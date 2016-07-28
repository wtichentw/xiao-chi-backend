filename="./link/$1"
exec < $filename

while read line
do
    echo $line # 一行一行印出內容
    wget -T 10 --tries=2 -P ./photo/$1 $line
done
