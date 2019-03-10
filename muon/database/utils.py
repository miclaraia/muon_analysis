
class StorageObject:

    def __init__(self, database, online=False):
        self.database = database
        self.storage = {}
        self.online = online

    @property
    def conn(self):
        return self.database.conn

    def save(self):
        pass


class StorageAttribute:

    def __init__(self, name):
        self.name = name

    def __set__(self, instance, value):
        obj = instance.storage[self.name]
        obj.set(value)

        if instance.online:
            instance.save()

    def __get__(self, instance, owner):
        return instance.storage[self.name].get()


class StoredAttribute:

    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.has_changed = False

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        self.has_changed = True

    def __str__(self):
        return '{}: {}, has_change: {}'.format(
            self.name, self.value, self.has_changed)

    def __repr__(self):
        return str(self)
