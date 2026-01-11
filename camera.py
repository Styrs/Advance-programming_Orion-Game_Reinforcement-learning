import Solar_System
import drawing
import UI_manager


class camera ():

    def __init__ (self,screen):
        
        self.maxwidth = screen.get_width()   #size of the screen at max
        self.maxheight = screen.get_height()
        
        self.camera_positionX = self.maxwidth // 2  #position of the camera at the center of the screen at the max size
        self.camera_positionY = self.maxheight // 2
        self.zoomed_width = self.maxwidth           #size in pixel of the screen at the current zoom level related to the max size screen
        self.zoomed_height = self.maxheight

        self.actual_width = self.maxwidth   #size of the map, always equal the the same resolution
        self.actual_height = self.maxheight

        self.zoom_level = 1
        self._recompute_limits()
        self.camera_centering()


    ## -------------------------------------------------------------------------------------------------------------------------------
    ## Zoming, mouse position conversion, camera movement 
    ## -------------------------------------------------------------------------------------------------------------------------------

    def zoom(self, zoom_type):
        """zoom_type > 0 => in, < 0 => out; updates limits and clamps center."""
        
        

        if zoom_type > 0 and self.zoom_level < 10.5: #zoom in
            self.zoom_level = self.zoom_level + 0.5
        elif zoom_type < 0 and self.zoom_level > 1: #zoom out
            self.zoom_level = self.zoom_level - 0.5
        else:
            return  # no change

        self.zoomed_width  = self.maxwidth  / self.zoom_level
        self.zoomed_height = self.maxheight / self.zoom_level

        self._recompute_limits()
        self.camera_centering()  # keep center valid after zoom

    

    def move_camera(self, dx_pixels, dy_pixels):
        """Drag/pan by screen-pixel deltas."""
        self.camera_positionX -= dx_pixels / self.zoom_level
        self.camera_positionY -= dy_pixels / self.zoom_level
        self.camera_centering()  # ensure we stay in bounds



    def _recompute_limits(self):
        """Recalculate what the camera can see and the legal center bounds."""
        self.half_width_camera  = (self.actual_width  / 2) / self.zoom_level
        self.half_height_camera = (self.actual_height / 2) / self.zoom_level

        self.min_x = self.half_width_camera
        self.max_x = self.maxwidth  - self.half_width_camera
        self.min_y = self.half_height_camera
        self.max_y = self.maxheight - self.half_height_camera



    def camera_centering(self):
        """Clamp the camera center inside the map boundaries."""
        if self.camera_positionX < self.min_x:
            self.camera_positionX = self.min_x
        elif self.camera_positionX > self.max_x:
            self.camera_positionX = self.max_x

        if self.camera_positionY < self.min_y:
            self.camera_positionY = self.min_y
        elif self.camera_positionY > self.max_y:
            self.camera_positionY = self.max_y



    def mouse_click_into_worldcord (self,x,y):
        x_worldcord = (x - self.actual_width / 2) / self.zoom_level + self.camera_positionX
        y_worldcord = (y - self.actual_height / 2) / self.zoom_level + self.camera_positionY
        print(f"Mouse world coordinates: ({x_worldcord}, {y_worldcord})")
        return (x_worldcord,y_worldcord)

    ## -------------------------------------------------------------------------------------------------------------------------------
    ## Clicking entities or UI elements
    ## -------------------------------------------------------------------------------------------------------------------------------

    def get_clicked_UI_elements (self,x,y):
        
        for ui_element in UI_manager.list_of_all_Ui_elements_visible:
            
            possible_clicked_element = ui_element.Get_ui_element_by_coordonate(x,y)
            if possible_clicked_element != False:
                return possible_clicked_element

        return False



    def get_clicked_entity (self,x,y,current_system):
        
        map_position = self.mouse_click_into_worldcord(x,y)
        entity = current_system.get_entities_from_coordinates(map_position[0],map_position[1])
        return entity



    def screen_got_clicked (self,x,y,current_system):

        entity = self.get_clicked_entity(x,y,current_system)
        if entity != None:


            if entity.category == "Planetoid":
                UI_manager.open_planet_ui(entity,self.maxwidth,self.maxheight)

            if entity.category == "Star":
                pass



            return entity
        elif entity == None:
            ui_element = self.get_clicked_UI_elements(x,y)
            if ui_element != False:
                return ui_element
            elif ui_element == False:
                return None



    ## -------------------------------------------------------------------------------------------------------------------------------
    ## Rendering function
    ## -------------------------------------------------------------------------------------------------------------------------------


    def System_rendering (self,screen,zoom_level,current_system):
        

        #draw Star
        for star in current_system.star_list:
            x = star.positionX
            y = star.positionY
            if abs(self.camera_positionX - x) < self.zoomed_width // 2 and abs(self.camera_positionY - y) < self.zoomed_height // 2:
                color_star= star.color
                x_rel = (x - self.camera_positionX) * self.zoom_level + self.actual_width / 2
                y_rel = (y - self.camera_positionY) * self.zoom_level + self.actual_height / 2
                drawing.draw_circle(screen,color_star,x_rel,y_rel,int(star.radius_pixel*self.zoom_level))

        #draw planets
        for planet in current_system.planet_list:
            x = planet.positionX
            y = planet.positionY
            if abs(self.camera_positionX - x) < self.zoomed_width // 2 and abs(self.camera_positionY - y) < self.zoomed_height // 2:
                color_planet = planet.color
                x_rel = (x - self.camera_positionX) * self.zoom_level + self.actual_width / 2
                y_rel = (y - self.camera_positionY) * self.zoom_level + self.actual_height / 2
                drawing.draw_circle(screen,color_planet,x_rel,y_rel,int(planet.radius_pixel*self.zoom_level))

    
    
    def Ui_rendering (self,screen):
        for ui_element in UI_manager.list_of_all_Ui_elements_visible:
            ui_element.draw_UI_element(screen)



    