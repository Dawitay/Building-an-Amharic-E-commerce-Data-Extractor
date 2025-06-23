from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv

# Load environment variables once
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')

# Function to read channels from a CSV file
def read_channels_from_csv(filename):
    channels = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            channels.append(row[0].strip())  # Ensure no extra spaces
    return channels

# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title  # Extract the channel's title
        async for message in client.iter_messages(entity, limit=10000):
            media_path = None
            if message.media and hasattr(message.media, 'photo'):
                # Create a unique filename for the photo
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                # Download the media to the specified directory if it's a photo
                #await client.download_media(message.media, media_path)
            
            # Write the channel title along with other data
            writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])
    except Exception as e:
        print(f"Error scraping channel {channel_username}: {e}")

# Initialize the client once
client = TelegramClient('scraping_session', api_id, api_hash)

async def main():
    await client.start()
    
    # Create a directory for media files if not already exists
    media_dir = 'photos'
    os.makedirs(media_dir, exist_ok=True)

    # Open the CSV file and prepare the writer (append mode)
    with open('../../data/raw_data/telegram_data.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Check if the file is empty to write headers
        file.seek(0, os.SEEK_END)  # Move to the end of the file
        if file.tell() == 0:
            writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])  # Include channel title in the header
        
        # Read channels from the CSV file
        channels = read_channels_from_csv('../../data/raw_data/channels_to_crawl.csv')
        
        # Iterate over channels and scrape data into the single CSV file
        for channel in channels:
            await scrape_channel(client, channel, writer, media_dir)
            print(f"Scraped data from {channel}")

with client:
    client.loop.run_until_complete(main())
