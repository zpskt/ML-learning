if __name__ == '__main__':
    import kagglehub

    # Download latest version
    path = kagglehub.competition_download('cifar-100-image-classification-challenge')

    print("Path to competition files:", path)