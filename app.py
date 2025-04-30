from flask import Flask, render_template, request, redirect, url_for, flash, session
import numpy as np
from option_game import Call, Put, RiskReversal, CallSpread, PutSpread, Straddle, Strangle, Portfolio, DiceSimulator

app = Flask(__name__)
app.secret_key = 'dice_options_game_secret_key'

# Create a global simulator that persists between requests
simulator = DiceSimulator()

@app.route('/')
def index():
    """Main page showing portfolio and game status"""
    return render_template('index.html', 
                          simulator=simulator,
                          portfolio_value=simulator.calculate_portfolio_value(),
                          rolls=simulator.rolls,
                          total=simulator.get_total() if simulator.rolls else 0,
                          Call=Call,
                          Put=Put,
                          RiskReversal=RiskReversal,
                          CallSpread=CallSpread,
                          PutSpread=PutSpread,
                          Straddle=Straddle,
                          Strangle=Strangle)

@app.route('/add_option', methods=['POST'])
def add_option():
    """Add an option to the portfolio"""
    option_type = request.form.get('option_type')
    quantity = int(request.form.get('quantity', 1))
    
    try:
        if option_type == 'call':
            strike = int(request.form.get('strike'))
            option = Call(strike)
        elif option_type == 'put':
            strike = int(request.form.get('strike'))
            option = Put(strike)
        elif option_type == 'risk_reversal':
            put_strike = int(request.form.get('put_strike'))
            call_strike = int(request.form.get('call_strike'))
            option = RiskReversal(put_strike, call_strike)
        elif option_type == 'call_spread':
            lower_strike = int(request.form.get('lower_strike'))
            higher_strike = int(request.form.get('higher_strike'))
            option = CallSpread(lower_strike, higher_strike)
        elif option_type == 'put_spread':
            higher_strike = int(request.form.get('higher_strike'))
            lower_strike = int(request.form.get('lower_strike'))
            option = PutSpread(higher_strike, lower_strike)
        elif option_type == 'straddle':
            strike = int(request.form.get('strike'))
            option = Straddle(strike)
        elif option_type == 'strangle':
            put_strike = int(request.form.get('put_strike'))
            call_strike = int(request.form.get('call_strike'))
            option = Strangle(put_strike, call_strike)
        else:
            flash('Invalid option type', 'error')
            return redirect(url_for('index'))
            
        simulator.add_to_portfolio(option, quantity)
        flash(f'Added {quantity} x {option} to portfolio', 'success')
        
    except ValueError as e:
        flash(f'Error: {str(e)}', 'error')
        
    return redirect(url_for('index'))

@app.route('/roll_die', methods=['POST'])
def roll_die():
    """Roll a single die"""
    value = request.form.get('value')
    if value:
        simulator.roll_die(int(value))
    else:
        simulator.roll_die()
    return redirect(url_for('index'))

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the simulator"""
    simulator.reset_rolls()
    # Create a new portfolio
    simulator.portfolio = Portfolio()
    flash('Game reset', 'success')
    return redirect(url_for('index'))

# @app.route('/set_spread', methods=['POST'])
# def set_spread():
#     """Set the bid-ask spread"""
#     spread = float(request.form.get('spread', 0.05))
#     simulator.set_spread(spread)
#     flash(f'Bid-ask spread set to {spread:.2%}', 'success')
#     return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
