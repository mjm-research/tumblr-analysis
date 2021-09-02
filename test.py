print('Hello my name is Brandon')
print('I live in Charlottesville')
print('Hello my name is Michelle')
print('I live in Connecticut')
print('Hello my name is Pepper')
print('I live where I want')

def say_name(name):
    print('Hello my name is ' + name)
    
def say_place(place):
    print("I live in " + place)

# DRY - Don't Repeat Yourself

print('I live in Connecticut')
print('I live in Charlottesville')
print('I live where I want')

say_name('Brandon')
say_place('Charlottesville')

say_name('Michelle')
say_place('Connecticut')

say_name('Pepper')
say_place('Where I want')

# objects - things that have qualities and can do things (adjectives and verbs)
# object here is a person -
# have a name, have a location, and can say both
# class - is a template for a object

class Person(object, name, location):
    def __init__(self):
        # when created, give our object some things
        self.name = name
        self.location = location
    
    def say_name(self):
        print('Hello my name is' + self.name)
    

person_one = Person('Brandon', 'Charlottesville')
person_two = Person('Michelle', 'Connecticut')
person_three = Person('Pepper', "Where I want")
        
person_one.name
person_three.location
person_one.say_name()
        
        
        
        




