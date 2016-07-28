filename='zongzhi.txt'
exec < $filename

while read line
do
    curl -O $line
    echo $line # 一行一行印出內容
done
