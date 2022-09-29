from typing import NamedTuple, List
from libzhifan import io

class ContactRegion(NamedTuple):
    """
    primary_verts/faces: List of 5 finger regions
        ordered from {thumb, index, middle, ring, pinky}
    secondary_verts/faces: List of 3 hand regions
        ordered from {thumb
    """
    primary_verts: List
    primary_faces: List
    secondary_verts: List
    secondary_faces: List


def get_contact_regions(path='weights/contact_regions.json'):
    contact_regions = io.read_json(path)
    return ContactRegion(**contact_regions)
