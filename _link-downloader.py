import urllib.response, urllib.request, sys, os, shutil, requests

category = sys.argv[1]

link_dir = os.path.join(os.getcwd(), 'link')
photo_dir = os.path.join(os.getcwd(), 'photo')
food_type_dir = os.path.join(photo_dir, category)

file_name = os.path.join(link_dir, category)

def ensure_dir_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)



ensure_dir_exist(link_dir)
ensure_dir_exist(photo_dir)
ensure_dir_exist(food_type_dir)

with open(file_name) as file:
    for i, line in enumerate(file):
        r = requests.get(line, stream=True, headers={'User-agent': 'Mozilla/5.0'})
        if r.status_code == 200:
            with open(os.path.join(food_type_dir, str(i)+'.jpg'), 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
                print ('success download %s.jpg' % str(i))
        else:
            print (r.status_code)
