from google_images_search import GoogleImagesSearch
from dotenv import load_dotenv
import os
import tempfile

def SearchImage(prompt, num_images=10):
  load_dotenv()

  gis = GoogleImagesSearch(os.environ['Google_API_KEY'], os.environ['Google_SearchEngine_ID'])
  _search_params = {
    'q': prompt,
    'num': num_images,
    # 'fileType': 'png',
    # 'rights': 'cc_publicdomain',
    'safe': 'active', ##
    'imgType': 'photo', ##
    # 'imgSize': 'large', ##
    'imgColorType': 'color' ##
  }

  gis.search(search_params=_search_params,
             path_to_dir=tempfile.gettempdir(),
             width=512, height=512,
             custom_image_name='reference_image')
  return gis.results()
