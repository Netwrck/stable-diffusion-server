## open sd.csv


import os
import shutil
import requests
from PIL import Image, ImageOps
from tqdm import tqdm
import imdb




def api_key():

    SAMPLE_URL = 'https://api.themoviedb.org/3/movie/76341?api_key={0}'
    try:
        while True:
            if os.path.exists("api.txt") and os.path.getsize("api.txt") > 0:
                f = open("api.txt", "r")
                key = f.read()
                req = requests.get(SAMPLE_URL.format(key))
                if req.status_code == 200:
                    print("\nAPI Key authentication successful.\n")
                    return key
                else:
                    pass

            print("\nTo register for an API key, Visit: https://www.themoviedb.org/account/signup")
            # get_key = input("API key required. Please enter the API key: ")
            get_key = "2e06a6a2d721d0886b346241e286dd7c"

            req = requests.get(SAMPLE_URL.format(get_key))
            if req.status_code == 200:
                f = open("api.txt", "w")
                f.write(get_key)
                f.close()
            else:
                print("\nInvalid API key: You must be granted a valid key.")
    except OSError:
        print("\nUnknown Error")

def search_movie(movie_name):

    try:
        ia = imdb.IMDb()
        user_input = movie_name
        print("\n", end="")
        movies = ia.search_movie(user_input)

        choices_list = []

        for i in movies:
            get_title = i['title']
            get_id = i.movieID
            get_year = ''

            try:
                get_year = i['year']
            except KeyError:
                pass
            p = ("{: <10}".format(str(get_id))+ get_title + " " + "(" + str(get_year) + ")")
            choices_list.append(p)

        # movie_list = questionary.select("Oh! there's alot. What did you mean? ", choices=choices_list).ask()
        if not choices_list:
            return None
        movie_list   = choices_list[0]
        get_id = movie_list.split()
        IMDB_ID = get_id[0]

        return IMDB_ID

    except KeyboardInterrupt:
        print("\nKeyboard Interrupted")
    except ValueError:
        print("\nUnknown movie name.")

def get_image_url(new_name):
    movie = search_movie(new_name)
    if not movie:
        return None
    URL = 'https://api.themoviedb.org/3/find/tt{0}?api_key={1}&language=en-US&external_source=imdb_id'.format(
        movie, api_key())

    req = requests.get(URL).json()
    for k, v in req.items():
        for i in v:
            for k, v in i.items():
                if k == 'poster_path':
                    if not v:
                        return None
                    image_url = 'http://image.tmdb.org/t/p/w500/' + v
                    return [image_url, v]
                # if i['poster_path'] is None:
                #     print("No Poster Found")
                #     quit()

# URL = 'https://api.themoviedb.org/3/find/tt{0}?api_key={1}&language=en-US&external_source=imdb_id'.format(search_movie(), api_key())
#
# req = requests.get(URL).json()

def remove_special_characters(input):
    return ''.join(e for e in input if e.isalnum() or e == ' ')


def get_image_url_imdb(search_query):
    search_path = f"https://api.themoviedb.org/3/search/movie?api_key=15d2ea6d0dc1d476efbca3eba2b9bbfb&query={search_query}"
    try:
        response = requests.get(search_path)
        data = response.json()
        if 'results' in data and data['results']:
            first_result = data['results'][0]
            if 'poster_path' in first_result and first_result['poster_path']:
                return ["http://image.tmdb.org/t/p/w500/" + first_result['poster_path'], first_result['poster_path']]
        print(f'no results for {search_query}')
        print(data)

    except Exception as e:
        print(f'err {e}')



def download_poster(new_name):
    fileextension = 'webp'
    name_no_comment = new_name.split("//")[0]
    name_no_col = new_name.split(":")[0]
    no_comment_no_special_chars = remove_special_characters(name_no_comment)
    no_special_chars = remove_special_characters(new_name)
    new_name = no_special_chars
    new_name = new_name.replace(" ", "_")
    file_save_path = f'full/{new_name}.{fileextension}'
    if os.path.exists(file_save_path):
        print("File already exists")
        return os.getcwd() + '/' + file_save_path
    # image_url = get_image_url(no_comment_no_special_chars)
    # image_url = get_image_url(name_no_comment)
    # image_url = get_image_url(name_no_col)
    # image_url = get_image_url_imdb(name_no_col)

    # image_url = get_image_url_imdb(no_comment_no_special_chars)
    # image_url = get_image_url_imdb(name_no_comment)
    image_url = get_image_url_imdb(name_no_col)
    if image_url is None:
        name_no_season = new_name.split("Season")[0]
        image_url = get_image_url_imdb(name_no_season)


    if image_url is None:
        print("\nNo poster found")
    else:
        returning_list = image_url
        url = returning_list[0]
        filename = returning_list[1]
        _response = requests.get(url).content
        file_size_request = requests.get(url, stream=True)
        file_size = int(file_size_request.headers['Content-Length'])
        block_size = 1024
        t = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading", ascii=True)
        filename_ = filename[1:]
        # fileextension = filename_.split(".")[1]


        # with open(file_save_path, 'wb') as f:
        #     for data in file_size_request.iter_content(block_size):
        #         t.update(len(data))
        #         f.write(data)
        # instead write to bytesio
        from io import BytesIO
        bytes = BytesIO()
        for data in file_size_request.iter_content(block_size):
            t.update(len(data))
            bytes.write(data)
        bytes.seek(0)
        image = Image.open(bytes)
        image.save(file_save_path, format="webp", quality=85)
        t.close()
        print("\nPoster downloaded successfully\n")
        return os.getcwd() + '/' + file_save_path

def convert(new_name):

    # this discussion helped me a lot: https://stackoverflow.com/questions/765736/using-pil-to-make-all-white-pixels-transparent

    icon = os.getcwd() + "/icons/" + new_name + ".png"
    bytes = download_poster(new_name)
    if bytes is None:
        return None
    # skip writing the smaller version todo webp
    # img = Image.open(bytes)
    # img = ImageOps.expand(img, (69, 0, 69, 0), fill=0)
    # img = ImageOps.fit(img, (300, 300)).convert("RGBA")
    #
    # datas = img.getdata()
    # newData = []
    # for item in datas:
    #     if item[0] == 0 and item[1] == 0 and item[2] == 0:
    #         newData.append((0, 0, 0, 0))
    #     else:
    #         newData.append(item)
    #
    # img.putdata(newData)
    # img.save(icon)
    # img.close()
    # print("Poster icon created successfully.")
    return icon


convert('planet of the apes')


import csv
from urllib.parse import quote

md_links = []
with open('sd.csv', 'r') as f:
    reader = csv.reader(f)
    converted = list(reader)
    for row in tqdm(converted[4911+5135:]):
        # convert to a link
        prompt = row[0]
        try:
            convert(prompt)
        except Exception as e:
            print(e)
            print(prompt)

# with open('sd.csv') as f:
#     csv_reader = csv.reader(f)
#     for row in csv_reader:
#         # print(row)
#         name = row[0]
#         name= name.split(" //")[0]
#         # get an image for the given name
#         #
#         # convert the image to a png
#
#         print(name)
