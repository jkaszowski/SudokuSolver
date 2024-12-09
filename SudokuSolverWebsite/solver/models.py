from django.db import models

class Photo(models.Model):
    image = models.ImageField(upload_to='uploads/')
    result = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"Photo {self.id}"