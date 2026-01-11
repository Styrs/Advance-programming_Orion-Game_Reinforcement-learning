import drawing 
import pygame
import Solar_System

list_of_all_Ui_elements_visible = [] ##super important list to initialize here first

##################################################################################################################################
## -------------------------------------------------------------------------------------------------------------------------------
## Ui function
## -------------------------------------------------------------------------------------------------------------------------------
##################################################################################################################################


def get_ui_element(element_class):
    """
    Returns the first UI element of a given class found in ui_list.
    If none is found, returns None.
    """
    for element in list_of_all_Ui_elements_visible:
        if isinstance(element, element_class):
            return element
    return None




##################################################################################################################################
## -------------------------------------------------------------------------------------------------------------------------------
## UI configuration, what sets of Ui element need to be visible at a given time
## -------------------------------------------------------------------------------------------------------------------------------
##################################################################################################################################
current_ui_configuration = None
clock = None
top_bar = None

def solar_map_ui_configuration (screen_width,screen_height):
    ## We will need to create a clock, a top bar. Maybe some menu on top left and bottom. 
    global current_ui_configuration 
    current_ui_configuration = "solar_map"
    
    global clock
    if clock == None:
        clock = Clock(screen_width,screen_height)
        list_of_all_Ui_elements_visible.append(clock)

    global top_bar
    if top_bar == None:
        
        top_bar = Top_bar(screen_width,screen_height)
        list_of_all_Ui_elements_visible.append(top_bar)
        top_bar.change_UI_mod_rendering(current_ui_configuration)
    else:
        
        top_bar.change_UI_mod_rendering(current_ui_configuration)


def open_planet_ui (planet,screen_width,screen_height):

    global current_ui_configuration 
    current_ui_configuration = "planet_mod"

    planet_ui = Planet_UI_window (screen_width,screen_height,planet)
    list_of_all_Ui_elements_visible.append(planet_ui)

    global top_bar
    
    top_bar.change_UI_mod_rendering(current_ui_configuration)
        

##################################################################################################################################
## -------------------------------------------------------------------------------------------------------------------------------
## UI Element classes
## -------------------------------------------------------------------------------------------------------------------------------
##################################################################################################################################




class UI_element():
    
    def __init__(self,
        x=0,y=0,width=0,height=0,*,
        surface=None,background_color=(0, 0, 0),form="rectangle",is_button=False,):
        
        self.screen_UI_positionX = x
        self.screen_UI_positionY = y
        self.UI_width = width
        self.UI_height = height
        self.surface = surface #the screen object 
        self.background_color = background_color
        self.form = form 
        self.is_button = is_button

        self.sub_Ui_object_list = []


## -------------------------------------------------------------------------------------------------------------------------------
## screen related functions
## -------------------------------------------------------------------------------------------------------------------------------

    def draw_UI_element (self,screen):
        if self.form == "rectangle":
            drawing.draw_rectangle(screen,self.background_color,self.screen_UI_positionX,self.screen_UI_positionY,self.UI_width,self.UI_height)
        if self.sub_Ui_object_list != []:
            for ui_object in self.sub_Ui_object_list:
                ui_object.draw_UI_object(screen)


    def Get_ui_element_by_coordonate (self,x,y):

        
        if x >= self.screen_UI_positionX and x <= self.screen_UI_positionX + self.UI_width and y >= self.screen_UI_positionY and y <= self.screen_UI_positionY + self.UI_height:
            
            
            if len(self.sub_Ui_object_list) == 0:
                return self
            elif len(self.sub_Ui_object_list) != 0:
                
                for sub_UI_object in self.sub_Ui_object_list:
                    if sub_UI_object.Get_sub_ui_object_by_coordonate(x,y) != False:
                       
                        return sub_UI_object
            return self
        return False
    

## -------------------------------------------------------------------------------------------------------------------------------
## interacting functions
## -------------------------------------------------------------------------------------------------------------------------------

    def update_UI_element (self):
        pass
    

    def close_UI_element (self):
        list_of_all_Ui_elements_visible.remove(self)


    def close_all_sub_UI_elem (self):
            
            for sub_UI_element in reversed(self.sub_Ui_object_list):
                print(f"et hop{len(self.sub_Ui_object_list)}")
                sub_UI_element.close_sub_UI_object()
               
## -------------------------------------------------------------------------------------------------------------------------------
## other functions
## -------------------------------------------------------------------------------------------------------------------------------

    def is_button_functional (self,sub_object_id):
        if self.is_button == True:
            print(f"Button with id {sub_object_id} clicked!")
            self.button_functionality(sub_object_id)


              
##################################################################################################################################
## -------------------------------------------------------------------------------------------------------------------------------
## Children of UI Element classes; specific type of UI element
## -------------------------------------------------------------------------------------------------------------------------------
##################################################################################################################################

## -------------------------------------------------------------------------------------------------------------------------------
## The clock UI element
## -------------------------------------------------------------------------------------------------------------------------------


class Clock(UI_element):

    def __init__(self,screen_width,screen_height):

        x = screen_width - (screen_width // 10)  # top-right corner
        y = 0
        width = screen_width // 10
        height = screen_height // 14
        background_color = (60, 100, 185)
        is_button = True

        super().__init__(x,y,width,height,background_color= background_color, is_button=is_button)
        
        
        
        

        for i in range (3):
            if i < 2:
                button = Button(self,sub_element_id = i,relative_UI_positionX = (i+1) * (self.UI_width // 4),
                                relative_UI_positionY = self.UI_height // 2,UI_width = self.UI_width // 4,
                                UI_height = self.UI_height // 2,object_color = (200,220,250))
                self.sub_Ui_object_list.append(button)
                if i == 0:
                    button.object_image = True
                    button.object_image_link = "Images/play_icon.png"
                elif i == 1:
                    button.object_image = True
                    button.object_image_link = "Images/pause_icon.png"

            else:
                text = TextObject(content = "00:00",font = "Arial",size = 20,color = (255,255,255))
                label_time = label(self,text = text,sub_element_id = i,relative_UI_positionX = self.UI_width // 4,
                                relative_UI_positionY = self.screen_UI_positionY,UI_width = self.UI_width // 2,
                                UI_height = self.UI_height //  2,object_color = (60,100,185))
                self.sub_Ui_object_list.append(label_time)





    def button_functionality (self,sub_object_id):
        print(f"Button with id {sub_object_id} clicked and do something!")
    
        ## to see on what sub_object we click to a for in the list of sub to see if we are on coordonates

        ## And for this object we have 3 sub : the Label with time, 2 button (play and pose) each have an id that will activate a specifc function

    def update_clock_time (self):
        self.sub_Ui_object_list[2].text = "12:34"  ## just an example to update the label text


## -------------------------------------------------------------------------------------------------------------------------------
## The top_bar UI element
## -------------------------------------------------------------------------------------------------------------------------------


class Top_bar(UI_element):

    def __init__(self,screen_width,screen_height):
        x = 0
        y = 0
        width = 9 * screen_width // 10
        height = screen_height // 15 
        background_color = (40, 70, 130)
        super().__init__(x,y,width,height,background_color=background_color,is_button=True)
        self.planet_mod = False
        self.is_top_left_menu_open = False
        
     
    def change_UI_mod_rendering (self,current_ui_configuration):
        
        


        if current_ui_configuration == "solar_map":
            self.close_all_sub_UI_elem()
            #Top left menu button
            menu_button = Button(self,0,0,0,self.UI_width//10,self.UI_height,object_color=(60,90,150))
            self.sub_Ui_object_list.append(menu_button)

            #label with the money amount 
            text_money_label = TextObject("money amount",font = "Arial",size = 20,color = (255,155,155))
            money_label = label(self,text_money_label,1,self.UI_width//5+5,self.UI_height/3,self.UI_width//5,self.UI_height/2,object_color=self.background_color)
            self.sub_Ui_object_list.append(money_label)

        elif current_ui_configuration == "planet_mod":
            self.close_all_sub_UI_elem()

            #Top left menu button
            menu_button = Button(self,0,0,0,self.UI_width//10,self.UI_height,object_color=(60,90,150),
                                 border_thickness=5,border_color=(175,220,255))
            self.sub_Ui_object_list.append(menu_button)


            #label with the money amount 
            text_money_label = TextObject("money amount",font = "Arial",size = 20,color = (255,155,155))
            money_label = label(self,text_money_label,1,self.UI_width//7+5,self.UI_height/3,self.UI_width//5,self.UI_height/2,object_color=self.background_color)
            self.sub_Ui_object_list.append(money_label)

            #label with the food amount 
            text_food_label = TextObject("food amount",font = "Arial",size = 20,color = (255,155,155))
            food_label = label(self,text_food_label,2,self.UI_width//7*2+5,self.UI_height/3,self.UI_width//5,self.UI_height/2,object_color=self.background_color)
            self.sub_Ui_object_list.append(food_label)

            #label with the metal amount 
            text_metal_label = TextObject("metal amount",font = "Arial",size = 20,color = (255,155,155))
            metal_label = label(self,text_metal_label,3,self.UI_width//7*3+5,self.UI_height/3,self.UI_width//5,self.UI_height/2,object_color=self.background_color)
            self.sub_Ui_object_list.append(metal_label)

            #label with the industrial good amount 
            text_industrial_good_label = TextObject("industrial good amount",font = "Arial",size = 20,color = (255,155,155))
            industrial_good_label = label(self,text_industrial_good_label,4,self.UI_width//7*4+5,self.UI_height/3,self.UI_width//5,self.UI_height/2,object_color=self.background_color)
            self.sub_Ui_object_list.append(industrial_good_label)

            #label with the consumption amount 
            text_consumption_good_label = TextObject("consumption goog amount",font = "Arial",size = 20,color = (255,155,155))
            consumption_good_label = label(self,text_consumption_good_label,5,self.UI_width//7*5+5,self.UI_height/3,self.UI_width//5,self.UI_height/2,object_color=self.background_color)
            self.sub_Ui_object_list.append(consumption_good_label)

            #label with the energy amount 
            text_energy_label = TextObject("energy amount",font = "Arial",size = 20,color = (255,155,155))
            energy_label = label(self,text_energy_label,6,self.UI_width//7*6+5,self.UI_height/3,self.UI_width//5,self.UI_height/2,object_color=self.background_color)
            self.sub_Ui_object_list.append(energy_label)

    def button_functionality (self,sub_object_id):
        print(f"Button with id {sub_object_id} clicked and do something!")
       
        
        top_left_menu = get_ui_element(Top_left_menu)

        if top_left_menu == None:
            # OPEN
            print("the top left menu is open")
            top_left_menu = Top_left_menu(self.UI_width, self.UI_height)
            list_of_all_Ui_elements_visible.append(top_left_menu)

        else:
            # CLOSE
            print("is he closed ?")
            top_left_menu.close_UI_element()
            
## -------------------------------------------------------------------------------------------------------------------------------
## The Planet UI element
## -------------------------------------------------------------------------------------------------------------------------------

class Planet_UI_window(UI_element):
    
    
    def __init__(self,screen_width,screen_height,planet_object):
        # Compute specific values for this UI element
        x = 5
        y = screen_height // 4
        width = screen_width // 4.5
        height = screen_height // 1.8
        background_color = (130, 135, 170)
        is_button = True
        

        super().__init__(x,y,width,height,background_color=background_color,is_button=is_button)
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.planet_object = planet_object

        close_button = Button(self,sub_element_id = 0,relative_UI_positionX = self.UI_width - 20,relative_UI_positionY = 0,
                              UI_width = 20,UI_height = 10,object_color = (255,0,0))
        self.sub_Ui_object_list.append(close_button)


    def button_functionality (self,sub_object_id):
        print(f"Button with id {sub_object_id} clicked and do something!")
        if sub_object_id == 0: # if 0 its the close button
            self.close_UI_element()
            solar_map_ui_configuration (self.screen_width,self.screen_height)
            
    
        ## to see on what sub_object we click to a for in the list of sub to see if we are on coordonates

        ## And for this object we have 3 sub : the Label with time, 2 button (play and pose) each have an id that will activate a specifc function


## -------------------------------------------------------------------------------------------------------------------------------
## The Top_left_menu UI element
## -------------------------------------------------------------------------------------------------------------------------------


class Top_left_menu(UI_element):


    def __init__(self,top_bar_width,top_bar_height):
        
        #number of button is important do change if we want to add button in that menu
        number_of_button = 1

        size_of_buttons = 30
        x=0
        y=top_bar_height
        width=top_bar_width//10
        height= size_of_buttons*number_of_button
        background_color=(60,90,150)

        super().__init__(x, y, width, height, background_color=background_color, is_button=True)

        

        text_button_commerce = TextObject("Commerce menu","Arial",size=20,color = (255,155,155))
        button_commerce = Button(self,0,0,0,width,size_of_buttons,
                                 object_color=background_color,is_text_surface=True,text=text_button_commerce)
        self.sub_Ui_object_list.append(button_commerce)

    def button_functionality (self,sub_object_id):
        print(f"Button with id {sub_object_id} clicked and do something!")
        if sub_object_id ==0:
            print("Commerce menu not yet coded")

##################################################################################################################################
## -------------------------------------------------------------------------------------------------------------------------------
## Element on the Ui element. Called sub_UI_object
## -------------------------------------------------------------------------------------------------------------------------------  
##################################################################################################################################



class sub_UI_object():

   
    def __init__(self,Master_UI_element,sub_element_id,relative_UI_positionX,relative_UI_positionY,UI_width,UI_height,*,
            object_color=None,is_object_image=False,object_image_link=None,is_text_surface=False,text=None,sub_element_interactivity=False,
            sub_element_type=None,border_thickness=0, border_color=None,border_top=True, border_right=True,border_bottom=True, border_left=True):

        self.Master_UI_element = Master_UI_element
        self.sub_element_id = sub_element_id
        
        self.screen_UI_positionX = Master_UI_element.screen_UI_positionX + relative_UI_positionX
        self.screen_UI_positionY = Master_UI_element.screen_UI_positionY + relative_UI_positionY

        self.relative_UI_positionX = relative_UI_positionX
        self.relative_UI_positionY = relative_UI_positionY
        self.UI_width = UI_width
        self.UI_height = UI_height

        self.border_thickness = border_thickness
        self.border_color = border_color
        self.border_top = border_top
        self.border_right = border_right
        self.border_bottom = border_bottom
        self.border_left = border_left
        
        self.is_object_image = is_object_image
        self.object_image_link = object_image_link
        self.object_color = object_color
        self.is_text_surface = is_text_surface

        if text != None:
            self.text = text.content
            self.font = text.font
            self.size = text.size
            self.color = text.color

        self.sub_element_interactivity = sub_element_interactivity
        self.sub_element_type = sub_element_type

## -------------------------------------------------------------------------------------------------------------------------------
## screen related functions
## -------------------------------------------------------------------------------------------------------------------------------
    def draw_UI_object (self,screen):
        if self.is_object_image == False:
            drawing.draw_rectangle(screen, self.object_color,self.screen_UI_positionX, self.screen_UI_positionY,self.UI_width, self.UI_height,
                border_thickness=self.border_thickness,border_color=self.border_color,border_top=self.border_top,
                border_right=self.border_right,border_bottom=self.border_bottom,border_left=self.border_left)

        elif self.is_object_image == True:
            image = pygame.image.load(self.object_image_link)
            image = pygame.transform.scale(image,(self.UI_width,self.UI_height))
            screen.blit (image,(self.screen_UI_positionX,self.screen_UI_positionY))



        if self.is_text_surface == True:
            drawing.draw_text(screen,self.screen_UI_positionX ,self.screen_UI_positionY ,self.text,self.size,self.color)


    def Get_sub_ui_object_by_coordonate (self,x,y):
       
        if x >= self.screen_UI_positionX and x <= self.screen_UI_positionX + self.UI_width and y >= self.screen_UI_positionY and y <= self.screen_UI_positionY + self.UI_height:
            
            
            if self.sub_element_type == "button":
                print("We are in get sub ui object by coordonate button section")
                self.Master_UI_element.is_button_functional(self.sub_element_id)
            
            return self
        return False
    

## -------------------------------------------------------------------------------------------------------------------------------
## Other functions
## -------------------------------------------------------------------------------------------------------------------------------


    def close_sub_UI_object (self):
        self.Master_UI_element.sub_Ui_object_list.remove(self)
        print(f"{self} has been erased from the list")
        


##################################################################################################################################
## -------------------------------------------------------------------------------------------------------------------------------
## Children of sub_UI_object classes; specific type of sub_UI_object
## -------------------------------------------------------------------------------------------------------------------------------
##################################################################################################################################


## -------------------------------------------------------------------------------------------------------------------------------
## Button sub UI object
## -------------------------------------------------------------------------------------------------------------------------------


class  Button(sub_UI_object):


    def __init__(
        self,Master_UI_element,sub_element_id,relative_UI_positionX,relative_UI_positionY,UI_width,UI_height,*,
        object_color = None,is_object_image = False,object_image_link = None,is_text_surface=False,text=None,
        border_thickness=0, border_color=None,border_top=True, border_right=True,border_bottom=True, border_left=True):
        
        type = "button"

        super().__init__(
                    Master_UI_element,sub_element_id,relative_UI_positionX,relative_UI_positionY,UI_width,UI_height,
                    object_color=object_color,is_object_image=is_object_image,object_image_link=object_image_link,is_text_surface=is_text_surface,
                    sub_element_interactivity=True,sub_element_type=type,text=text,
                    border_thickness=border_thickness,border_color=border_color,
                    border_top=border_top,border_right=border_right,border_bottom=border_bottom,border_left=border_left)
                
        
        print(f"Button with coordinates ({self.screen_UI_positionX}, {self.screen_UI_positionY}) created!")


## -------------------------------------------------------------------------------------------------------------------------------
## Label sub UI object
## -------------------------------------------------------------------------------------------------------------------------------

    
class label(sub_UI_object):
    
 def __init__(
        self,Master_UI_element,text,sub_element_id,relative_UI_positionX,relative_UI_positionY,UI_width,UI_height,*,
        object_color=None,is_object_image=False,object_image_link=None,
        border_thickness=0, border_color=None,border_top=True, border_right=True,border_bottom=True, border_left=True):

        type = "label"
        text_surface = True


        super().__init__(
            Master_UI_element,sub_element_id,relative_UI_positionX,relative_UI_positionY,UI_width,UI_height,
            object_color=object_color,is_object_image=is_object_image,object_image_link=object_image_link,
            is_text_surface=text_surface,text=text,sub_element_interactivity=False,sub_element_type=type,
            border_thickness=border_thickness,border_color=border_color,
            border_top=border_top,border_right=border_right,border_bottom=border_bottom,border_left=border_left

            )        

        
        
        



class TextObject ():
    #this allows to create text objects with all the necessary attributes

    def __init__(self,content,font,size,color):
        self.content = content
        self.font = font
        self.size = size
        self.color = color

