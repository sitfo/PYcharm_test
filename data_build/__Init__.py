
import data_extract
import datadownload

if __name__ == "__main__":
    """
    Main function to download and extract data from the web.
    You can change the content of the main function to download and extract data by changing the author's page number in datadownload.py
    the data will be downloaded and extracted in the data folder.
    """
    datadownload.main()
    data_extract.main()
