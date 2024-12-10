import argparse
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image

def fetch_europeana_context(api_key, search_query, max_result, **kwargs):
    """
    Function to fetch Europeana context.
    """
    try:
        # Europeana API endpoint
        url = "https://api.europeana.eu/record/v2/search.json"
        params = {
            'wskey': api_key,
            'query': search_query,
            'start': kwargs.get('start', 1),  # Start from the first result
            'rows': max_result,  # Number of results per page
            # 'df' : kwargs.get(
        }

        # Optional filters based on CLI arguments
        if kwargs.get('type'):
            params['type'] = kwargs['type']  # Media type (IMAGE, VIDEO, etc.)
        if kwargs.get('reusability'):
            params['reusability'] = kwargs['reusability']  # Reusability level (open, restricted, etc.)
        if kwargs.get('country'):
            params['country'] = kwargs['country']  # Filter by country
        if kwargs.get('language'):
            params['language'] = kwargs['language']  # Filter by language (ISO 639-1 code)
        if kwargs.get('sort'):
            params['sort'] = kwargs['sort']  # Sorting (relevance, timestamp_created, etc.)
        if kwargs.get('facet'):
            params['facet'] = kwargs['facet']  # Facet-specific results
        if kwargs.get('profile'):
            params['profile'] = kwargs['profile']  # Response profile (minimal, standard, rich)
        if kwargs.get('qf'):
            params['qf'] = kwargs['qf']  # Query refinement filter
        if kwargs.get('colour_palette'):
            params['colour_palette'] = kwargs['colour_palette']  # Filter by image color (hex code)
        if kwargs.get('timestamp'):
            params['timestamp_created'] = kwargs['timestamp']  # Filter by timestamp range
        if kwargs.get('distance'):
            params['distance'] = kwargs['distance']  # Geographic range filter

        # Fetch the Europeana API
        response = requests.get(url, params=params, timeout=10)
 
        response.raise_for_status()
        data = response.json()
        print(response.json()['itemsCount'])
        exit()
        if data.get('totalResults', 0) == 0:
            return None, None, None, None, None, "No results found."
        
        item = data.get('items', [])[0]
        description = item.get('dcDescription', ['No description available'])[0]
        title = item.get('title', ['No Title'])[0]
        provider_url = item.get('guid', None)
        if not provider_url:
            return title, description, None, None, None, "Provider URL not found."

        # Fetch result metadata
        response = requests.get(provider_url, timeout=10)
        response.raise_for_status()
        metadata = fetch_metadata(response.content)
        image = fetch_image(response.content)

        return title, description, provider_url, metadata, image, None

    except requests.exceptions.RequestException as e:
        return None, None, None, None, None, f"Error fetching Europeana context: {e}"

def fetch_metadata(html_content):
    """
    Function to extract metadata from HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    h3_texts = [h3.get_text(strip=True) for h3 in soup.find_all('h3')]
    li_texts = []
    for ul in soup.find_all('ul'):
        li_items = [li.get_text(strip=True) for li in ul.find_all('li')]
        concatenated_li_text = ', '.join(li_items)
        if concatenated_li_text not in ['Home, Collections, Stories, Share your collections, Log in / Join', '']:
            li_texts.append(concatenated_li_text)
    metadata = '\n'
    for str1, str2 in zip(h3_texts, li_texts):
        metadata += str1 + ': ' + str2 + ',\n'
    return metadata

def fetch_image(html_content):
    """
    Function to extract the image download URL from HTML content using BeautifulSoup and download the image.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    download_link = soup.find('a', class_='download-button')
    if download_link and download_link.has_attr('href'):
        image_url = download_link['href']
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return img
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    else:
        print("Download link not found in the HTML content.")
        return None


def parse_args():
    """Returns: Command-line arguments"""
    parser = argparse.ArgumentParser('Europeana-api, command-line interface (CLI)')
    parser.add_argument('--key', type=str, help='Europeana API key (required)', required=True)
    parser.add_argument('--search-query', type=str, help='Search keyword(s) (required)', required=True)
    parser.add_argument('--max-results', type=int, default=6, help='Maximum number of results to return (default: 10)')
    # TODO Optional parameters with defaults
    # parser.add_argument('--data-provider', type=str, default=None, help='Filter by data provider (default: None)')
    # parser.add_argument('--institute', type=str, default=None, help='Filter by contributing institution (default: None)')
    # parser.add_argument('--type', type=str, choices=['IMAGE', 'VIDEO', 'TEXT', 'SOUND'], default=None, help='Filter by media type (default: None)')
    # parser.add_argument('--reusability', type=str, choices=['open', 'restricted', 'permission'], default=None, help='Filter by reusability level (default: None)')
    # parser.add_argument('--country', type=str, default=None, help='Filter by country (default: None)')
    # parser.add_argument('--language', type=str, default=None, help='Filter results by language (ISO 639-1 code) (default: None)')
    # parser.add_argument('--start', type=int, default=1, help='Pagination start (default: 1)')
    # parser.add_argument('--sort', type=str, choices=['relevance', 'timestamp_created', 'timestamp_update'], default='relevance', help='Sort results (default: relevance)')
    # parser.add_argument('--profile', type=str, choices=['minimal', 'standard', 'rich'], default='minimal', help='Specify the profile for response data (default: minimal)')
    # parser.add_argument('--facet', type=str, default=None, help='Request specific facets (default: None)')
    # parser.add_argument('--qf', type=str, default=None, help='Query refinement filter (default: None)')
    # parser.add_argument('--colour-palette', type=str, default=None, help='Filter by image color (hex format, e.g., #FF5733) (default: None)')
    # parser.add_argument('--timestamp', type=str, default=None, help='Filter by creation or update timestamp (default: None)')
    # parser.add_argument('--distance', type=str, default=None, help='Filter by geographic distance (default: None)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    title, description, provider_url, metadata, image, _ = fetch_europeana_context(
        api_key=args.key,
        search_query=args.search_query,
        max_result=args.max_results,

        # TODO Optional parameters with defaults
        # data_provider=args.data_provider,
        # institute=args.institute,
        # type=args.type,
        # reusability=args.reusability,
        # country=args.country,
        # language=args.language,
        # start=args.start,
        # sort=args.sort,
        # profile=args.profile,
        # facet=args.facet,
        # qf=args.qf,
        # colour_palette=args.colour_palette,
        # timestamp=args.timestamp,
        # distance=args.distance
    )

    print("=" * 80)
    print(f"API Key: {args.key}")
    print(f"Search Query: {args.search_query}")
    print(f"Title: {title}")
    print(f"Description: {description}")
    print(f"Provider URL: {provider_url}")
    print(f"Metadata: {metadata}")
    print("=" * 80)