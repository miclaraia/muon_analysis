from tqdm import tqdm

import muon.project.panoptes as pan


def main():
    project_a = pan.Project.find(5918)
    project_b = pan.Project.find(9284)

    group1 = 75029
    group2 = 75083

    pan.Uploader.client()

    # for group in [group1, group2]:
        # print(group)
        # subject_set = pan.SubjectSet.find(group)
    # project_b.links.subject_sets = [group1, group2]
    # print(project_b.links.subject_sets)
        # print(subject_set, subject_set.display_name)
        # subject_set.links.project = project_b
        # subject_set.save()
    # project_b.save()

    for group in [group1, group2]:
        old = pan.SubjectSet.find(group)
        new = pan.SubjectSet()

        print(old, new)

        new.display_name = old.display_name
        new.links.project = project_b
        new.save()

        # import code
        # code.interact(local={**globals(), **locals()})

        subjects = []
        for s in tqdm(old.subjects):
            new.add([s])
        # print(subjects)
        # new.links.subjects = subjects

        new.save()


if __name__ == '__main__':
    main()
