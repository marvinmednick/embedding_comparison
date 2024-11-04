import requests
from bs4 import BeautifulSoup

text_sources = [
    {
        'title': "The Time Machine by H.G. Wells",
        'file':  'time_machine_hg_wells.txt',
        'url': "https://www.gutenberg.org/files/35/35-h/35-h.htm"
    },
    {
        'title': "The Yellow Wallpaper by Charlotte Perkins Gilman",
        'file':  'yellow_wallpper_cp_gilman.txt',
        'url':  "https://www.gutenberg.org/files/1952/1952-h/1952-h.htm",
    },
    {
        'title': "The Metamorphosis by Franz Kafka",
        'file': 'metamorphosis_kafka',
        'url':  "https://www.gutenberg.org/files/5200/5200-h/5200-h.htm",
    },
]


for entry in text_sources:
    print(f"Downloading {entry['title']}")
    url = entry['url']
    output_file = entry['file']
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the main content
    content = soup.find('body').find_all(['p', 'h1', 'h2', 'h3'])

    # Extract and join the text
    text = ' '.join([para.get_text() for para in content])

    # Remove extra whitespace
    text = ' '.join(text.split())

    with open(output_file, "w") as output:
        output.write(text)
