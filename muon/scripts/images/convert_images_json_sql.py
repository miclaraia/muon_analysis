import click
import json

import muon.project.sql_images as sql_images
import muon.project.images as json_images


def convert_image(image):
    kwargs = {
        'image_id': image.id,
        'group_id': image.group,
        'subjects': image.subjects,
        'metadata': image.metadata,
        'zoo_id': image.zoo_id
    }
    return sql_images.Image(**kwargs)


def convert_image_group(images):
    _images = {i: convert_image(images.images[i]) for i in images.images}
    return sql_images.ImageGroup(
        group_id=images.group,
        images=_images,
        image_size=images.size,
        image_width=images.image_dim,
        description=images.description,
        permutations=images.permutations)

@click.group(invoke_without_command=True)
@click.argument('json_in')
@click.argument('sql_out')
def main(json_in, sql_out):
    json_storage = json_images.Images._list_groups()
    sql_storage = sql_images.SQLImages(sql_out)

    print(json_images.Images._list_groups())
    for group in json_images.Images._list_groups():
        images = json_images.Images.load_group(group, fname=json_in)
        sql_storage._groups[images.group] = convert_image_group(images)

    sql_storage.save()
    print(images.__dict__)
    print(images)


if __name__ == '__main__':
    main()

