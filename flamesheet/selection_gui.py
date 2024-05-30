# # -*- coding: utf-8 -*-
# """
# Created on Thu Oct  6 14:48:06 2022

# @author: laaltenburg
# """

# import tkinter as tk
# import os

# def selection_gui():
    
#     def on_selection(value):
#         global choice
#         choice = value  # store the user's choice
#         records_list(value)
#         # quit_me()
    
#     def quit_me():
#         print('quit')
#         root.quit()
#         root.destroy()
    
#     def records_list(value):
        
#         quit_me()
        
#     root = tk.Tk()
#     root.protocol("WM_DELETE_WINDOW", quit_me)
    
#     # Create a Tkinter variable
#     tkvar = tk.StringVar(root)
    
#     # options
#     main_dir = "Y:/"
#     projects = os.listdir(main_dir)
#     projects = [project for project in projects if not '.exp' in project]
    
#     # records_choices = 
#     tkvar.set("flamesheet_2d_nonreact_day5") # set the default option
    
#     project_menu = tk.OptionMenu(root, tkvar, *projects, command=on_selection)
#     records_menu = tk.OptionMenu(root, tkvar, *projects, command=on_selection)
#     tk.Label(root, text="Choose an experimental data").grid(row=0, column=0)
#     project_menu.grid(row=1, column=0)
#     records_menu.grid(row=2, column=0)
    
    
#     root.mainloop()
    
#     # Do whatever you want with the user's choice after closing the window
#     print('You have chosen %s' % choice)
    
#     print(choice)
    
#     return choice

# #%% Main
# if __name__ == "__main__":
    
#     selection_gui()
#%% 2
# import sys
import os

# if sys.version_info[0] >= 3:
#     import tkinter as tk
#     from tkinter import ttk
# else:
#     import Tkinter as tk


# class App(tk.Frame):

#     def __init__(self, master):
        
#         tk.Frame.__init__(self, master)
        
        
        
#         main_dir = "Y:/"
#         projects = os.listdir(main_dir)
#         projects = [project for project in projects if not '.exp' in project]
        
#         self.dict = {}

        
#         for project in projects:
            
#             records = os.listdir(main_dir + project + "/")
#             records = [record for record in records if not '.set' in record]
            
#             self.dict[project] = records

#         self.variable_project = tk.StringVar(self)
#         self.variable_record = tk.StringVar(self)

#         self.variable_project.trace('w', self.update_options)

#         self.optionmenu_projects = tk.OptionMenu(self, self.variable_project, *self.dict.keys())
#         self.optionmenu_records = tk.OptionMenu(self, self.variable_record, '')
        
#         self.title = tk.Label(self, text="Choose experimental data")
#         self.variable_project.set('Projects')
#         self.variable_record.set('Recordings')
        
#         # button
#         options = {'padx': 5, 'pady': 5}
#         self.button = ttk.Button(self, text='Click Me')
#         self.button['command'] = self.button_clicked
#         self.button.pack(**options)
        
#         self.title.pack()
#         self.optionmenu_projects.pack()
#         self.optionmenu_records.pack()
#         self.pack()
    
    
#     # def button_clicked(self):
#     #     self.quit_me()
#     #     print(self.dict[self.variable_project.get()])


        
#     def update_options(self, *args):
#         records = self.dict[self.variable_project.get()]
#         self.variable_record.set(records[0])

#         menu = self.optionmenu_records['menu']
#         menu.delete(0, 'end')

#         for record in records:
#             menu.add_command(label=record, command=lambda recording=record: self.variable_record.set(recording))


# if __name__ == "__main__":
#     root = tk.Tk()
#     app = App(root)
#     app.mainloop()    
    

#%% 3
import tkinter as tk
from tkinter import *
  
# Class for tkinter window
  
  
class Window():
    
    def __init__(self):
  
        # Creating the tkinter Window
        self.root = Tk()
        # self.root.geometry("200x100")
  
        
        main_dir = "Y:/"
        projects = os.listdir(main_dir)
        projects = [project for project in projects if not '.exp' in project]
        
        self.dict = {}
        
        for project in projects:
            
            records = os.listdir(main_dir + project + "/")
            self.records = [record for record in records if not '.set' in record]
            
            self.dict[project] = self.records

        self.variable_project = tk.StringVar(self.root)
        self.variable_record = tk.StringVar(self.root)
        
        def update_options(self, *args):
            self.records = self.dict[self.variable_project.get()]
            self.variable_record.set(self.records[0])

            menu = self.optionmenu_records['menu']
            menu.delete(0, 'end')

            for record in records:
                menu.add_command(label=record, command=lambda recording=record: self.variable_record.set(recording))
                
        self.variable_project.trace('w', update_options)

        self.optionmenu_projects = tk.OptionMenu(self.root, self.variable_project, *self.dict.keys())
        self.optionmenu_records = tk.OptionMenu(self.root, self.variable_record, '')
        
        title = Label(self.root, text="Choose experimental data")
        self.variable_project.set('Projects')
        self.variable_record.set('Recordings')
    
        title.pack()
        self.optionmenu_projects.pack()
        self.optionmenu_records.pack()
        
        # Button for closing
        exit_button = Button(self.root, text="Exit", command=self.root.destroy)
        exit_button.pack(pady=20)
        
        self.root.mainloop()
  
  
# Running test window
test = Window()