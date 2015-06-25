class DataClass:
  def __init__(self):
    self.title = ""
    self.masker = None
    self.signal = None
    self.H = None
    self.image = None
    self.comments = ''
  
  def update(self,title =' ', masker = None, signal = None,H = None, image = None,
		 comments = ' '):
    self.title = title
    self.masker = masker
    self.signal = signal
    self.H = H
    self.image = image
    self.comments = comments