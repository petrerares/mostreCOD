/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Petre
 */
public class Container {
    private int amount;
    private int max;
    
    public Container(){
        this.amount=0;
        this.max=100;
        
    }
    
    public int contains(){
        return this.amount;
    }
    
    public void add(int add){
        if(add>=0){
          this.amount=this.amount+add;
          if(this.amount>this.max){
            this.amount=this.max;
          }  
        }
        
    }
    
    public int move(int move){
        if(move>=0){
            if(move>this.amount){
                move=this.amount;
            }
            
            this.amount=this.amount-move;
            return move;
        }
        
        return 0;
    }
    
    public void remove(int remove){
        if(remove>=0){
            if(remove>this.amount){
                remove=this.amount;
            }
            
            this.amount=this.amount-remove;
            
        }
        
        
    }
    
    public String toString(){
        return this.amount+"/"+this.max;
    }
}
