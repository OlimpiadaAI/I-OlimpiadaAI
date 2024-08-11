import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageDraw
from IPython.display import display, HTML
import json

from utils.general import _load_images_from_folder


def _update_frame(i, img_container, images):
    img_container.set_array(images[i])
    return (img_container,)


def _add_tracks_to_image(pil_image, tracks, frame_numeber):
    if frame_numeber < 2:
        return pil_image
    first_cup_positions, second_cup_positions, third_cup_positions = tracks['first_cup'], tracks['second_cup'], tracks[
        'third_cup']
    draw = ImageDraw.Draw(pil_image)
    for i in range(1, frame_numeber):
        if first_cup_positions[i - 1] is not None and first_cup_positions[i] is not None:
            draw.line([first_cup_positions[i - 1], first_cup_positions[i]], fill='red', width=5)
        if second_cup_positions[i - 1] is not None and second_cup_positions[i] is not None:
            draw.line([second_cup_positions[i - 1], second_cup_positions[i]], fill='blue', width=5)
        if third_cup_positions[i - 1] is not None and third_cup_positions[i] is not None:
            draw.line([third_cup_positions[i - 1], third_cup_positions[i]], fill='green', width=5)
    return pil_image


def display_video(src, first_preds=[], second_preds=[], third_preds=[],
                        tracks={}, rescale=1, FINAL_EVALUATION_MODE=True):
    """
    Generates a video from a sequence of images, with object tracks overlaid.

    Parameters:
    src (str or list): Either a path to the folder containing the images (str), or a list of PIL Image objects.
    tracks (dict): Dictionary containing the object tracks to be displayed on the images. 
                   The keys should be object identifiers ("first_cup", "second_cup", "third_cup") and the values
                   should be sequences of positions (list of tuples with coordinates) e.g. [(0.323,13.454), (56.23,1.332), ...].
                   Each trace should be the same length as the number of images in the folder or list.
    rescale (float): Factor by which to rescale the video. For example, a rescale factor of 0.5 would reduce 
                     the size of the video to half of the original size of the images.

    Returns:
    HTML: An HTML object containing the generated video.
    """
    """
    Generuje wideo z sekwencji obrazów z nałożonymi śladami obiektów.

    Parametry:
    src (str lub lista): Ścieżka do folderu zawierającego obrazy (str) lub lista obiektów obrazów PIL.
    first_preds (list): Lista przewidywanych pozycji (x,y) dla pierwszego obiektu w każdej klatce.
    second_preds (list): Lista przewidywanych pozycji (x,y) dla drugiego obiektu w każdej klatce.
    third_preds (list): Lista przewidywanych pozycji (x,y) dla trzeciego obiektu w każdej klatce.
    tracks (dict): Słownik zawierający ślady obiektów do wyświetlenia na obrazach.
                   Klucze powinny być identyfikatorami obiektów ("first_cup", "second_cup", "third_cup"), a wartości
                   powinny być sekwencjami pozycji (lista krotek z współrzędnymi), np. [(0.323,13.454), (56.23,1.332), ...].
                   Każdy ślad powinien mieć długość równą liczbie obrazów w folderze lub liście.
    rescale (float): Współczynnik skalowania wideo. Na przykład współczynnik skalowania 0.5 zmniejszy
                     rozmiar wideo do połowy oryginalnego rozmiaru obrazów.

    Zwraca:
    HTML: Obiekt HTML zawierający wygenerowane wideo.
    """
    if FINAL_EVALUATION_MODE:
        return None

    if isinstance(src, list):
        images = src
    else:
        images = _load_images_from_folder(src)
        
    if tracks:
        # add tracks
        images = [_add_tracks_to_image(image, tracks, frame_numeber) for frame_numeber, image in enumerate(images)]

    # plot a prediction dot on the image, color of the track
    for i, pred in enumerate(first_preds):
        if pred is not None:
            draw = ImageDraw.Draw(images[i])
            draw.ellipse([pred[0] - 5, pred[1] - 5, pred[0] + 5, pred[1] + 5], fill='red')

    for i, pred in enumerate(second_preds):
        if pred is not None:
            draw = ImageDraw.Draw(images[i])
            draw.ellipse([pred[0] - 5, pred[1] - 5, pred[0] + 5, pred[1] + 5], fill='blue')
    
    for i, pred in enumerate(third_preds):
        if pred is not None:
            draw = ImageDraw.Draw(images[i])
            draw.ellipse([pred[0] - 5, pred[1] - 5, pred[0] + 5, pred[1] + 5], fill='green')


    img_width, img_height = images[0].size
    fig_width, fig_height = img_width / plt.rcParams['figure.dpi'], img_height / plt.rcParams['figure.dpi']

    fig, ax = plt.subplots(figsize=(fig_width * rescale, fig_height * rescale))
    ax.axis('off')  # This hides the axes

    img_container = plt.imshow(images[0], animated=True)

    ani = FuncAnimation(fig, _update_frame, frames=len(images), fargs=(img_container, images), blit=True, repeat=False)

    plt.close(fig)

    return HTML(ani.to_jshtml())
