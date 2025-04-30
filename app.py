from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import numpy as np
from option_game import Call, Put, RiskReversal, CallSpread, PutSpread, Straddle, Strangle, Portfolio, DiceSimulator

app = Flask(__name__)
app.secret_key = 'dice_options_game_secret_key'

# Create a global simulator that persists between requests
simulator = DiceSimulator()

@app.route('/')
def index():
    """Main page showing portfolio and game status"""
    # Calculate portfolio value
    portfolio_value = simulator.calculate_portfolio_value()
    
    # Calculate PNL if we have completed two rolls
    total_pnl, position_pnls = simulator.calculate_portfolio_pnl() if len(simulator.rolls) >= 2 else (None, [])
    
    return render_template('index.html', 
                          simulator=simulator,
                          portfolio_value=portfolio_value,
                          rolls=simulator.rolls,
                          total=simulator.get_total() if simulator.rolls else 0,
                          total_pnl=total_pnl,
                          position_pnls=position_pnls,
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
    entry_price = request.form.get('entry_price')
    
    # Convert entry_price to float if provided
    if entry_price and entry_price.strip():
        entry_price = float(entry_price)
    else:
        entry_price = None
    
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
            
        simulator.add_to_portfolio(option, quantity, entry_price)
        
        price_info = f" at price {entry_price:.2f}" if entry_price is not None else ""
        flash(f'Added {quantity} x {option}{price_info} to portfolio', 'success')
        
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

@app.route('/option_analytics', methods=['POST'])
def option_analytics():
    """Get analytics for an option before adding to portfolio"""
    option_type = request.form.get('option_type')
    
    try:
        # Create the option object based on form data
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
            return jsonify({'error': 'Invalid option type'}), 400
        
        # Calculate analytics
        # fair_value = simulator.calculate_option_value(option)
        # delta = option.delta()
        # vega = option.vega()
        
        # Calculate delta-neutral quantity (negative of inverse delta)
        analytics = simulator.get_option_analytics(option)
        fair_value, delta, vega, delta_neutral_quantity = analytics['fair_value'], analytics['delta'], analytics['vega'], analytics['delta_neutral_quantity']
        
        return jsonify({
            'option_name': str(option),
            'fair_value': round(fair_value, 4),
            'delta': round(delta, 4),
            'vega': round(vega, 4),
            'delta_neutral_quantity': round(delta_neutral_quantity)
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
