# spam_hunter
Exctract the archive of spams in Euron-spam in that same folder

Processed emails with extracted features of TF-IDF are saved to file processed_datasets.npz and loaded each time to reduce compuations needed from computing features matrices each time. If you need to process features again or add new features the class instantiation in "__main__.py" file needed to take a False as a last argument. 