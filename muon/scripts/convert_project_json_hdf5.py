import click
import json

import muon.project.hdf_images as h_images
import muon.project.images as j_images


def convert_image(image):
    kwargs = {
        'id_': image.id,
        'group': image.group,
        'subjects': image.subjects,
        'metadata': image.metadata,
        'zoo_id': image.zoo_id
    }
    return h_images.Image(**kwargs)


def convert_image_group(images):
    _images = {i: convert_image(images.images[i]) for i in images.images}
    out = h_images.ImageGroup(images.group, _images)
    out.metadata(images.metadata())
    return out

@click.group(invoke_without_command=True)
@click.argument('json_in')
@click.argument('hdf_out')
def main(json_in, hdf_out):
    groups = j_images.Images._list_groups()

    with open(json_in, 'r') as file:
        data = json.load(open(json_in, 'r'))
        new_images = h_images.HDFImages.new(hdf_out)
        new_images.next_id = data['next_id']
        new_images.next_group = data['next_group']
        del data

    print(j_images.Images._list_groups())
    for group in j_images.Images._list_groups():
        images = j_images.Images.load_group(group, fname=json_in)
        new_images._groups[images.group] = convert_image_group(images)

    new_images.save()
    print(images.__dict__)
    print(images)


if __name__ == '__main__':
    main()

