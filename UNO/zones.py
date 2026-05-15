def extract_zones(img):
    """
    Isolates and returns the 5 main regions from the board: 
    Player 1, Player 2, Player 3, Player 4, and the Middle.
    """
    p1 = img[1750:2662, 500:3600]
    p2 = img[200:2400, 2600:4000]
    p3 = img[0:1000, 500:3200]
    p4 = img[300:2500, 0:1300]
    mid = img[850:1850, 1000:3000]
    
    return p1, p2, p3, p4, mid