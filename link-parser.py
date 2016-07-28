# usage: python google-image-link-parser.py QUERY_TEXT REQUIRE_LINK_NUMBER OUTPUT_FILE_NAME
import urllib.request, urllib.parse, json, sys, os

link_dir = os.path.join(os.getcwd(), 'link')

api_key = 'AIzaSyD8EMpCHrq2ck8IxsXdyA2FcYJ-eSQoTy0'
cx_id = '015465916556109681566:pflt7bx_ta8'
api_url = 'https://www.googleapis.com/customsearch/v1?'

keyword = sys.argv[1]
req_link_num = int(sys.argv[2])
output_file = sys.argv[3]

cur_link_num = 0
links = []

def parse_json_to_link_list(json):
    for item in json["items"]:
        links.append(item["link"])

while cur_link_num < req_link_num:
    # get response from google custom search api
    encode_keyword = urllib.parse.quote(keyword)
    query = 'searchType=image&filtType=jpeg&q=%s&key=%s&cx=%s&start=%s' % (encode_keyword, api_key, cx_id, cur_link_num+1)
    response = urllib.request.urlopen(api_url + query)

    # load http response to json
    result = json.loads(response.read().decode('utf-8'))

    # parse json and add to links
    parse_json_to_link_list(result)

    # add count to cur_link_num
    cur_link_num += int(result["queries"]["request"][0]["count"])

# write to file
log = open(os.path.join(link_dir, output_file), "w")
for link in links:
    log.write(link + '\n')
log.close()


