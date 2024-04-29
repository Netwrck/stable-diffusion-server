def detect_too_bumpy(pil_image):
    # get color histogram
    hist = pil_image.histogram()
    r = hist[0:256]
    g = hist[256:512]
    b = hist[512:768]
    # count the pixels in the bottom 10 and top 10 of hist
    bottom = sum(r[:10]) + sum(g[:10]) + sum(b[:10])
    top = sum(r[-10:]) + sum(g[-10:]) + sum(b[-10:])

    # over 30px of deviation is variation/spike from old value

    # convert to grayscale with the formula
    # gray = 0.299*r + 0.587*g + 0.114*b
    # and get the histogram
    gray = [0]*256
    grayscale_img = pil_image.convert('L')
    gray_hist = grayscale_img.histogram()
    # deviation in the grayscale histogram
    deviations = 0
    # deviation is a spike in the histogram over 30px
    current_value = 0
    previous_value = 0
    for i in range(256):
        current_value = gray_hist[i]
        previous_value = gray_hist[i-1] if i > 0 else 0
        # iv deviation is over 20px
        if abs(current_value - previous_value) > 50:
            deviations += 1
            current_value = 0
    print(bottom, top, deviations)
    # return bottom > 30 or top > 30 or deviations > 30 # weird image
    return bottom > 30000 and top > 30000 and deviations > 80 # weird image
