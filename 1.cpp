#include<iostream>
#include<vector>
using namespace std;
int main(){
    int a;
    cin>>a;
    vector<int> arr(a);
    for(int b=0;b<a;b++){
        cin>>arr[b];
    }
    int c;
    int num=0;
    cin>>c;
    for(int b=0;b<a;b++){
        if (c==arr[b]){
            num++;
        }
    }
    cout<<num<<endl;
}