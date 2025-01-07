import requests

# URL of the file to download
url = "https://dumps.wikimedia.org/rowiki/20240601/rowiki-20240601-pages-articles-multistream.xml.bz2"

# Define the output file path
output_file = "rowiki-20240601-pages-articles-multistream.xml.bz2"

# Download the file
response = requests.get(url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    # Write the content to a file
    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"File downloaded and saved as {output_file}")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")
