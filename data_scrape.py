import os
import re
import requests
from bs4 import BeautifulSoup


def scrape_lyrics():
    base_url = "https://www.lyrics.com"
    artist_url = f"{base_url}/artist/The-1975/2688689"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }

    if not os.path.exists("data"):
        os.makedirs("data")

    try:
        response = requests.get(artist_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        song_links = soup.find_all("td", class_="tal qx")
        if not song_links:
            print("No song links found on the page.")
            return

        for song_link in song_links:
            a_tag = song_link.find("a")
            if a_tag and a_tag.has_attr("href"):
                song_page_url = f"{base_url}{a_tag['href']}"

                try:
                    song_response = requests.get(song_page_url, headers=headers)
                    song_response.raise_for_status()
                    song_soup = BeautifulSoup(song_response.text, "html.parser")
                    song_title = None
                    song_title_element = song_soup.find("h1", class_="lyric-title")
                    if song_title_element:
                        song_title = song_title_element.text.strip()
                    else:
                        song_title_from_link = a_tag.text.strip()
                        if song_title_from_link:
                            song_title = song_title_from_link
                        else:
                            title_tag = song_soup.find("title")
                            if title_tag:

                                title_text = title_tag.text.strip()
                                match = re.match(r"^(.*?) Lyrics by The 1975", title_text)
                                if match:
                                    song_title = match.group(1).strip()
                                else:
                                    song_title = title_text

                    if not song_title:
                        print(f"Could not find title for {song_page_url}")
                        continue

                    sanitized_title = re.sub(r'[\\/*?:"<>|]', "", song_title)

                    lyrics_element = song_soup.find("pre", id="lyric-body-text")
                    if not lyrics_element:
                        print(f"Could not find lyrics for '{song_title}' (from {song_page_url})")
                        continue

                    lyrics = lyrics_element.text.strip()

                    file_path = os.path.join("data", f"{sanitized_title}.txt")
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(lyrics)

                    print(f"Successfully scraped lyrics for '{song_title}'")

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching song page {song_page_url}: {e}")
                except Exception as e:
                    print(f"An error occurred while processing {song_page_url}: {e}")

        print("\nAll lyrics scraped successfully!")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching artist page {artist_url}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    scrape_lyrics()