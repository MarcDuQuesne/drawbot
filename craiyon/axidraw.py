from pyaxidraw import axidraw   # import module

ad = axidraw.AxiDraw()          # Initialize class
ad.interactive()                # Enter interactive context

           # Process changes to options

if not ad.connect():            # Open serial port to AxiDraw;
    quit()                      #   Exit, if no connection.
# Change some options, just to show how we do so:
ad.options.units = 0        # set working units back to inches.
ad.options.speed_pendown = 75     # set pen-down speed to fast
ad.options.pen_rate_lower = 10 # Set pen down very slowly
ad.options.pen_pos_up = 80         # Height of pen when raised (0-100). Default 60
ad.options.pen_pos_down = 30       # Height of pen when lowered (0-100). Default 30
ad.update()      

# ad.pen_lifts(1)                 # Set pen lift height to 1 inch
ad.penup()
# ad.pendown()

ad.plot_run()
ad.plot_cleanup()


                                # Absolute moves follow:
# ad.moveto(1, 1)                 # Pen-up move to (1 inch, 1 inch)
# ad.lineto(2, 1)                 # Pen-down move, to (2 inch, 1 inch)
# ad.moveto(0, 0)                 # Pen-up move, back to origin.
ad.disconnect()                 # Close serial port to AxiDraw

